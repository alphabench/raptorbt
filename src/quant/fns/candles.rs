//! Candlestick patterns: the 61 TA-Lib `CDL*` functions. Each returns 0, +100 or -100, with
//! +-80 and +-200 where TA-Lib has them (engulfing and harami touching on one side, the
//! confirming bar of hikkake).
//!
//! A pattern is a small `eval` over the last few bars. What "long", "short", "near" or "equal"
//! mean comes from [`CandleSettings`], TA-Lib's table of eleven settings (a range type, an
//! averaging period and a factor). The settings are explicit data held by the kernel: there
//! is no global state. Expressions always run on [`CandleSettings::TALIB`]; a Rust caller (the
//! changed-settings replay test) builds a node with its own table through
//! [`make_with_settings`].
//!
//! The warm-up of a pattern is its TA-Lib lookback: the bars the shape needs plus the longest
//! averaging period among the settings it reads, so it depends on the settings.
//!
//! Averages are running totals, as in TA-Lib: one total per (setting, bar offset) a pattern
//! declares, started with an ascending sum when the first output is due and then updated by
//! adding the newest range and removing the oldest. That keeps the arithmetic, and so every
//! tie on a comparison, identical to the reference. A NaN anywhere in the lookback window
//! gives NaN and drops the totals; the first clean window afterwards rebuilds them, so a NaN
//! never poisons them.

use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    regs, Domain, Family, FnDef, InputDef, Invariance, Memory, NanRule, OutKind, ParamDef,
    ParamKind, ParamVal, Reference, Relation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeType {
    RealBody,
    HighLow,
    Shadows,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Setting {
    pub range: RangeType,
    /// Bars averaged; 0 means the range of the bar itself.
    pub period: usize,
    pub factor: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Set {
    BodyLong,
    BodyVeryLong,
    BodyShort,
    BodyDoji,
    ShadowLong,
    ShadowVeryLong,
    ShadowShort,
    ShadowVeryShort,
    Near,
    Far,
    Equal,
}

const N_SETS: usize = 11;

/// TA-Lib's candle settings table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandleSettings {
    s: [Setting; N_SETS],
}

const fn setting(range: RangeType, period: usize, factor: f64) -> Setting {
    Setting { range, period, factor }
}

impl CandleSettings {
    /// TA-Lib's defaults, in the order of [`Set`].
    pub const TALIB: Self = Self {
        s: [
            setting(RangeType::RealBody, 10, 1.0), // BodyLong
            setting(RangeType::RealBody, 10, 3.0), // BodyVeryLong
            setting(RangeType::RealBody, 10, 1.0), // BodyShort
            setting(RangeType::HighLow, 10, 0.1),  // BodyDoji
            setting(RangeType::RealBody, 0, 1.0),  // ShadowLong
            setting(RangeType::RealBody, 0, 2.0),  // ShadowVeryLong
            setting(RangeType::Shadows, 10, 1.0),  // ShadowShort
            setting(RangeType::HighLow, 10, 0.1),  // ShadowVeryShort
            setting(RangeType::HighLow, 5, 0.2),   // Near
            setting(RangeType::HighLow, 5, 0.6),   // Far
            setting(RangeType::HighLow, 5, 0.05),  // Equal
        ],
    };

    pub fn get(&self, set: Set) -> Setting {
        self.s[set as usize]
    }

    /// The same table with one setting replaced.
    pub fn with(mut self, set: Set, setting: Setting) -> Self {
        self.s[set as usize] = setting;
        self
    }
}

impl Default for CandleSettings {
    fn default() -> Self {
        Self::TALIB
    }
}

use Set::{
    BodyDoji as BD, BodyLong as BL, BodyShort as BS, Equal as EQ, Far as FR, Near as NR,
    ShadowLong as SL, ShadowShort as SS, ShadowVeryLong as SVL, ShadowVeryShort as SVS,
};

fn range_of(b: &[[f64; 4]], t: RangeType, j: usize) -> f64 {
    let [o, h, l, c] = b[j];
    match t {
        RangeType::RealBody => (c - o).abs(),
        RangeType::HighLow => h - l,
        RangeType::Shadows => (h - c.max(o)) + (c.min(o) - l),
    }
}

/// The bars a pattern sees. `j` counts bars back from the current one (0).
pub struct Cx<'a> {
    b: &'a [[f64; 4]],
    set: &'a CandleSettings,
    /// The index into `tot` of each declared (setting, offset), `NO_AVG` for one not declared.
    slot: &'a [[u8; MAX_OFF]; N_SETS],
    tot: &'a [f64],
    pen: f64,
}

/// Bar offsets a pattern may read an average at, and the marker of an undeclared pair.
const MAX_OFF: usize = 8;
const NO_AVG: u8 = u8::MAX;

impl Cx<'_> {
    fn o(&self, j: usize) -> f64 {
        self.b[j][0]
    }
    fn h(&self, j: usize) -> f64 {
        self.b[j][1]
    }
    fn l(&self, j: usize) -> f64 {
        self.b[j][2]
    }
    fn c(&self, j: usize) -> f64 {
        self.b[j][3]
    }
    fn rb(&self, j: usize) -> f64 {
        (self.c(j) - self.o(j)).abs()
    }
    fn us(&self, j: usize) -> f64 {
        self.h(j) - self.c(j).max(self.o(j))
    }
    fn ls(&self, j: usize) -> f64 {
        self.c(j).min(self.o(j)) - self.l(j)
    }
    fn hl(&self, j: usize) -> f64 {
        self.h(j) - self.l(j)
    }
    /// +1 for a white candle (close at or above open), -1 for a black one.
    fn col(&self, j: usize) -> i32 {
        if self.c(j) >= self.o(j) {
            1
        } else {
            -1
        }
    }
    fn top(&self, j: usize) -> f64 {
        self.c(j).max(self.o(j))
    }
    fn bot(&self, j: usize) -> f64 {
        self.c(j).min(self.o(j))
    }
    /// The real body of `a` lies wholly above that of `b`.
    fn rb_gap_up(&self, a: usize, b: usize) -> bool {
        self.bot(a) > self.top(b)
    }
    fn rb_gap_dn(&self, a: usize, b: usize) -> bool {
        self.top(a) < self.bot(b)
    }
    /// The whole candle `a` lies above candle `b`.
    fn gap_up(&self, a: usize, b: usize) -> bool {
        self.l(a) > self.h(b)
    }
    fn gap_dn(&self, a: usize, b: usize) -> bool {
        self.h(a) < self.l(b)
    }

    /// The setting's average range for the bar `off` back. A pattern declares every
    /// (setting, offset) it reads in its spec; an undeclared one is a bug in the spec.
    #[inline(always)]
    fn avg(&self, set: Set, off: usize) -> f64 {
        let s = self.set.get(set);
        let x = if s.period > 0 {
            let k = self.slot[set as usize][off];
            assert!(k != NO_AVG, "undeclared average {set:?} at {off}");
            self.tot[usize::from(k)] / s.period as f64
        } else {
            range_of(self.b, s.range, off)
        };
        s.factor * x / if s.range == RangeType::Shadows { 2.0 } else { 1.0 }
    }
}

// ---------------------------------------------------------------------------- patterns
// In TA-Lib's alphabetical order. `c.x(j)` is bar `j` back from the current one.

fn cdl2crows(c: &Cx) -> i32 {
    if c.col(2) == 1
        && c.rb(2) > c.avg(BL, 2)
        && c.col(1) == -1
        && c.rb_gap_up(1, 2)
        && c.col(0) == -1
        && c.o(0) < c.o(1)
        && c.o(0) > c.c(1)
        && c.c(0) > c.o(2)
        && c.c(0) < c.c(2)
    {
        -100
    } else {
        0
    }
}

fn cdl3blackcrows(c: &Cx) -> i32 {
    if c.col(3) == 1
        && c.col(2) == -1
        && c.ls(2) < c.avg(SVS, 2)
        && c.col(1) == -1
        && c.ls(1) < c.avg(SVS, 1)
        && c.col(0) == -1
        && c.ls(0) < c.avg(SVS, 0)
        && c.o(1) < c.o(2)
        && c.o(1) > c.c(2)
        && c.o(0) < c.o(1)
        && c.o(0) > c.c(1)
        && c.h(3) > c.c(2)
        && c.c(2) > c.c(1)
        && c.c(1) > c.c(0)
    {
        -100
    } else {
        0
    }
}

fn cdl3inside(c: &Cx) -> i32 {
    if c.rb(2) > c.avg(BL, 2)
        && c.rb(1) <= c.avg(BS, 1)
        && c.top(1) < c.top(2)
        && c.bot(1) > c.bot(2)
        && ((c.col(2) == 1 && c.col(0) == -1 && c.c(0) < c.o(2))
            || (c.col(2) == -1 && c.col(0) == 1 && c.c(0) > c.o(2)))
    {
        -c.col(2) * 100
    } else {
        0
    }
}

fn cdl3linestrike(c: &Cx) -> i32 {
    if c.col(3) == c.col(2)
        && c.col(2) == c.col(1)
        && c.col(0) == -c.col(1)
        && c.o(2) >= c.bot(3) - c.avg(NR, 3)
        && c.o(2) <= c.top(3) + c.avg(NR, 3)
        && c.o(1) >= c.bot(2) - c.avg(NR, 2)
        && c.o(1) <= c.top(2) + c.avg(NR, 2)
        && ((c.col(1) == 1
            && c.c(1) > c.c(2)
            && c.c(2) > c.c(3)
            && c.o(0) > c.c(1)
            && c.c(0) < c.o(3))
            || (c.col(1) == -1
                && c.c(1) < c.c(2)
                && c.c(2) < c.c(3)
                && c.o(0) < c.c(1)
                && c.c(0) > c.o(3)))
    {
        c.col(1) * 100
    } else {
        0
    }
}

fn cdl3outside(c: &Cx) -> i32 {
    if (c.col(1) == 1 && c.col(2) == -1 && c.c(1) > c.o(2) && c.o(1) < c.c(2) && c.c(0) > c.c(1))
        || (c.col(1) == -1
            && c.col(2) == 1
            && c.o(1) > c.c(2)
            && c.c(1) < c.o(2)
            && c.c(0) < c.c(1))
    {
        c.col(1) * 100
    } else {
        0
    }
}

fn cdl3starsinsouth(c: &Cx) -> i32 {
    if c.col(2) == -1
        && c.col(1) == -1
        && c.col(0) == -1
        && c.rb(2) > c.avg(BL, 2)
        && c.ls(2) > c.avg(SL, 2)
        && c.rb(1) < c.rb(2)
        && c.o(1) > c.c(2)
        && c.o(1) <= c.h(2)
        && c.l(1) < c.c(2)
        && c.l(1) >= c.l(2)
        && c.ls(1) > c.avg(SVS, 1)
        && c.rb(0) < c.avg(BS, 0)
        && c.ls(0) < c.avg(SVS, 0)
        && c.us(0) < c.avg(SVS, 0)
        && c.l(0) > c.l(1)
        && c.h(0) < c.h(1)
    {
        100
    } else {
        0
    }
}

fn cdl3whitesoldiers(c: &Cx) -> i32 {
    if c.col(2) == 1
        && c.us(2) < c.avg(SVS, 2)
        && c.col(1) == 1
        && c.us(1) < c.avg(SVS, 1)
        && c.col(0) == 1
        && c.us(0) < c.avg(SVS, 0)
        && c.c(0) > c.c(1)
        && c.c(1) > c.c(2)
        && c.o(1) > c.o(2)
        && c.o(1) <= c.c(2) + c.avg(NR, 2)
        && c.o(0) > c.o(1)
        && c.o(0) <= c.c(1) + c.avg(NR, 1)
        && c.rb(1) > c.rb(2) - c.avg(FR, 2)
        && c.rb(0) > c.rb(1) - c.avg(FR, 1)
        && c.rb(0) > c.avg(BS, 0)
    {
        100
    } else {
        0
    }
}

fn cdlabandonedbaby(c: &Cx) -> i32 {
    if c.rb(2) > c.avg(BL, 2) && c.rb(1) <= c.avg(BD, 1) && c.rb(0) > c.avg(BS, 0) {
        if c.col(2) == -1
            && c.col(0) == 1
            && c.c(0) > c.c(2) + c.rb(2) * c.pen
            && c.gap_dn(1, 2)
            && c.gap_up(0, 1)
        {
            return 100;
        }
        if c.col(2) == 1
            && c.col(0) == -1
            && c.c(0) < c.c(2) - c.rb(2) * c.pen
            && c.gap_up(1, 2)
            && c.gap_dn(0, 1)
        {
            return -100;
        }
    }
    0
}

fn cdladvanceblock(c: &Cx) -> i32 {
    if c.col(2) == 1
        && c.col(1) == 1
        && c.col(0) == 1
        && c.c(0) > c.c(1)
        && c.c(1) > c.c(2)
        && c.o(1) > c.o(2)
        && c.o(1) <= c.c(2) + c.avg(NR, 2)
        && c.o(0) > c.o(1)
        && c.o(0) <= c.c(1) + c.avg(NR, 1)
        && c.rb(2) > c.avg(BL, 2)
        && c.us(2) < c.avg(SS, 2)
        && ((c.rb(1) < c.rb(2) - c.avg(FR, 2) && c.rb(0) < c.rb(1) + c.avg(NR, 1))
            || c.rb(0) < c.rb(1) - c.avg(FR, 1)
            || (c.rb(0) < c.rb(1)
                && c.rb(1) < c.rb(2)
                && (c.us(0) > c.avg(SS, 0) || c.us(1) > c.avg(SS, 1)))
            || (c.rb(0) < c.rb(1) && c.us(0) > c.avg(SL, 0)))
    {
        -100
    } else {
        0
    }
}

fn cdlbelthold(c: &Cx) -> i32 {
    if c.rb(0) > c.avg(BL, 0)
        && ((c.col(0) == 1 && c.ls(0) < c.avg(SVS, 0))
            || (c.col(0) == -1 && c.us(0) < c.avg(SVS, 0)))
    {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdlbreakaway(c: &Cx) -> i32 {
    if c.rb(4) > c.avg(BL, 4)
        && c.col(4) == c.col(3)
        && c.col(3) == c.col(1)
        && c.col(1) == -c.col(0)
        && ((c.col(4) == -1
            && c.rb_gap_dn(3, 4)
            && c.h(2) < c.h(3)
            && c.l(2) < c.l(3)
            && c.h(1) < c.h(2)
            && c.l(1) < c.l(2)
            && c.c(0) > c.o(3)
            && c.c(0) < c.c(4))
            || (c.col(4) == 1
                && c.rb_gap_up(3, 4)
                && c.h(2) > c.h(3)
                && c.l(2) > c.l(3)
                && c.h(1) > c.h(2)
                && c.l(1) > c.l(2)
                && c.c(0) < c.o(3)
                && c.c(0) > c.c(4)))
    {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdlclosingmarubozu(c: &Cx) -> i32 {
    if c.rb(0) > c.avg(BL, 0)
        && ((c.col(0) == 1 && c.us(0) < c.avg(SVS, 0))
            || (c.col(0) == -1 && c.ls(0) < c.avg(SVS, 0)))
    {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdlconcealbabyswall(c: &Cx) -> i32 {
    if c.col(3) == -1
        && c.col(2) == -1
        && c.col(1) == -1
        && c.col(0) == -1
        && c.ls(3) < c.avg(SVS, 3)
        && c.us(3) < c.avg(SVS, 3)
        && c.ls(2) < c.avg(SVS, 2)
        && c.us(2) < c.avg(SVS, 2)
        && c.rb_gap_dn(1, 2)
        && c.us(1) > c.avg(SVS, 1)
        && c.h(1) > c.c(2)
        && c.h(0) > c.h(1)
        && c.l(0) < c.l(1)
    {
        100
    } else {
        0
    }
}

fn cdlcounterattack(c: &Cx) -> i32 {
    if c.col(1) == -c.col(0)
        && c.rb(1) > c.avg(BL, 1)
        && c.rb(0) > c.avg(BL, 0)
        && c.c(0) <= c.c(1) + c.avg(EQ, 1)
        && c.c(0) >= c.c(1) - c.avg(EQ, 1)
    {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdldarkcloudcover(c: &Cx) -> i32 {
    if c.col(1) == 1
        && c.rb(1) > c.avg(BL, 1)
        && c.col(0) == -1
        && c.o(0) > c.h(1)
        && c.c(0) > c.o(1)
        && c.c(0) < c.c(1) - c.rb(1) * c.pen
    {
        -100
    } else {
        0
    }
}

fn cdldoji(c: &Cx) -> i32 {
    if c.rb(0) <= c.avg(BD, 0) {
        100
    } else {
        0
    }
}

fn cdldojistar(c: &Cx) -> i32 {
    if c.rb(1) > c.avg(BL, 1)
        && ((c.col(1) == 1 && c.rb_gap_up(0, 1)) || (c.col(1) == -1 && c.rb_gap_dn(0, 1)))
        && c.rb(0) <= c.avg(BD, 0)
    {
        -c.col(1) * 100
    } else {
        0
    }
}

fn cdldragonflydoji(c: &Cx) -> i32 {
    if c.rb(0) <= c.avg(BD, 0) && c.us(0) < c.avg(SVS, 0) && c.ls(0) > c.avg(SVS, 0) {
        100
    } else {
        0
    }
}

/// +-100 when the body engulfs the previous one strictly on both sides, +-80 when it ties on
/// one side.
fn cdlengulfing(c: &Cx) -> i32 {
    if c.col(0) == 1
        && c.col(1) == -1
        && ((c.c(0) >= c.o(1) && c.o(0) < c.c(1)) || (c.c(0) > c.o(1) && c.o(0) <= c.c(1)))
    {
        return if c.c(0) > c.o(1) && c.o(0) < c.c(1) { 100 } else { 80 };
    }
    if c.col(0) == -1
        && c.col(1) == 1
        && ((c.o(0) >= c.c(1) && c.c(0) < c.o(1)) || (c.o(0) > c.c(1) && c.c(0) <= c.o(1)))
    {
        return if c.o(0) > c.c(1) && c.c(0) < c.o(1) { -100 } else { -80 };
    }
    0
}

fn cdleveningdojistar(c: &Cx) -> i32 {
    if c.rb(2) > c.avg(BL, 2)
        && c.col(2) == 1
        && c.rb(1) <= c.avg(BD, 1)
        && c.rb_gap_up(1, 2)
        && c.rb(0) > c.avg(BS, 0)
        && c.col(0) == -1
        && c.c(0) < c.c(2) - c.rb(2) * c.pen
    {
        -100
    } else {
        0
    }
}

fn cdleveningstar(c: &Cx) -> i32 {
    if c.rb(2) > c.avg(BL, 2)
        && c.col(2) == 1
        && c.rb(1) <= c.avg(BS, 1)
        && c.rb_gap_up(1, 2)
        && c.rb(0) > c.avg(BS, 0)
        && c.col(0) == -1
        && c.c(0) < c.c(2) - c.rb(2) * c.pen
    {
        -100
    } else {
        0
    }
}

fn cdlgapsidesidewhite(c: &Cx) -> i32 {
    if ((c.rb_gap_up(1, 2) && c.rb_gap_up(0, 2)) || (c.rb_gap_dn(1, 2) && c.rb_gap_dn(0, 2)))
        && c.col(1) == 1
        && c.col(0) == 1
        && c.rb(0) >= c.rb(1) - c.avg(NR, 1)
        && c.rb(0) <= c.rb(1) + c.avg(NR, 1)
        && c.o(0) >= c.o(1) - c.avg(EQ, 1)
        && c.o(0) <= c.o(1) + c.avg(EQ, 1)
    {
        if c.rb_gap_up(1, 2) {
            100
        } else {
            -100
        }
    } else {
        0
    }
}

fn cdlgravestonedoji(c: &Cx) -> i32 {
    if c.rb(0) <= c.avg(BD, 0) && c.ls(0) < c.avg(SVS, 0) && c.us(0) > c.avg(SVS, 0) {
        100
    } else {
        0
    }
}

fn cdlhammer(c: &Cx) -> i32 {
    if c.rb(0) < c.avg(BS, 0)
        && c.ls(0) > c.avg(SL, 0)
        && c.us(0) < c.avg(SVS, 0)
        && c.bot(0) <= c.l(1) + c.avg(NR, 1)
    {
        100
    } else {
        0
    }
}

fn cdlhangingman(c: &Cx) -> i32 {
    if c.rb(0) < c.avg(BS, 0)
        && c.ls(0) > c.avg(SL, 0)
        && c.us(0) < c.avg(SVS, 0)
        && c.bot(0) >= c.h(1) - c.avg(NR, 1)
    {
        -100
    } else {
        0
    }
}

/// Harami: +-100 when the small body is strictly inside the long one, +-80 when it touches.
fn harami(c: &Cx, small: f64) -> i32 {
    if c.rb(1) > c.avg(BL, 1) && c.rb(0) <= small {
        if c.top(0) < c.top(1) && c.bot(0) > c.bot(1) {
            return -c.col(1) * 100;
        }
        if c.top(0) <= c.top(1) && c.bot(0) >= c.bot(1) {
            return -c.col(1) * 80;
        }
    }
    0
}

fn cdlharami(c: &Cx) -> i32 {
    harami(c, c.avg(BS, 0))
}

fn cdlharamicross(c: &Cx) -> i32 {
    harami(c, c.avg(BD, 0))
}

fn cdlhighwave(c: &Cx) -> i32 {
    if c.rb(0) < c.avg(BS, 0) && c.us(0) > c.avg(SVL, 0) && c.ls(0) > c.avg(SVL, 0) {
        c.col(0) * 100
    } else {
        0
    }
}

/// Hikkake and its modified form mark a pattern bar and look for a close beyond it within
/// three bars. TA-Lib carries that as state; it is a function of the last four bars (an older
/// pattern bar can no longer be confirmed), so it is rederived from them at every bar and
/// nothing is held between bars. `found(j)` is the pattern on the bar `j` back.
fn hikkake_scan(c: &Cx, found: impl Fn(&Cx, usize) -> i32) -> i32 {
    let mut res = 0;
    let mut idx: Option<usize> = None;
    for j in (0..=3).rev() {
        let r = found(c, j);
        if r != 0 {
            res = r;
            idx = Some(j);
            if j == 0 {
                return res;
            }
        } else if let Some(p) = idx {
            if p - j <= 3 && ((res > 0 && c.c(j) > c.h(p + 1)) || (res < 0 && c.c(j) < c.l(p + 1)))
            {
                if j == 0 {
                    return res + if res > 0 { 100 } else { -100 };
                }
                idx = None;
            }
        }
    }
    0
}

fn cdlhikkake(c: &Cx) -> i32 {
    hikkake_scan(c, |c, j| {
        if c.h(j + 1) < c.h(j + 2)
            && c.l(j + 1) > c.l(j + 2)
            && ((c.h(j) < c.h(j + 1) && c.l(j) < c.l(j + 1))
                || (c.h(j) > c.h(j + 1) && c.l(j) > c.l(j + 1)))
        {
            if c.h(j) < c.h(j + 1) {
                100
            } else {
                -100
            }
        } else {
            0
        }
    })
}

fn cdlhikkakemod(c: &Cx) -> i32 {
    hikkake_scan(c, |c, j| {
        if c.h(j + 2) < c.h(j + 3)
            && c.l(j + 2) > c.l(j + 3)
            && c.h(j + 1) < c.h(j + 2)
            && c.l(j + 1) > c.l(j + 2)
        {
            if c.h(j) < c.h(j + 1)
                && c.l(j) < c.l(j + 1)
                && c.c(j + 2) <= c.l(j + 2) + c.avg(NR, j + 2)
            {
                return 100;
            }
            if c.h(j) > c.h(j + 1)
                && c.l(j) > c.l(j + 1)
                && c.c(j + 2) >= c.h(j + 2) - c.avg(NR, j + 2)
            {
                return -100;
            }
        }
        0
    })
}

fn cdlhomingpigeon(c: &Cx) -> i32 {
    if c.col(1) == -1
        && c.col(0) == -1
        && c.rb(1) > c.avg(BL, 1)
        && c.rb(0) <= c.avg(BS, 0)
        && c.o(0) < c.o(1)
        && c.c(0) > c.c(1)
    {
        100
    } else {
        0
    }
}

fn cdlidentical3crows(c: &Cx) -> i32 {
    if c.col(2) == -1
        && c.ls(2) < c.avg(SVS, 2)
        && c.col(1) == -1
        && c.ls(1) < c.avg(SVS, 1)
        && c.col(0) == -1
        && c.ls(0) < c.avg(SVS, 0)
        && c.c(2) > c.c(1)
        && c.c(1) > c.c(0)
        && c.o(1) <= c.c(2) + c.avg(EQ, 2)
        && c.o(1) >= c.c(2) - c.avg(EQ, 2)
        && c.o(0) <= c.c(1) + c.avg(EQ, 1)
        && c.o(0) >= c.c(1) - c.avg(EQ, 1)
    {
        -100
    } else {
        0
    }
}

fn cdlinneck(c: &Cx) -> i32 {
    if c.col(1) == -1
        && c.rb(1) > c.avg(BL, 1)
        && c.col(0) == 1
        && c.o(0) < c.l(1)
        && c.c(0) <= c.c(1) + c.avg(EQ, 1)
        && c.c(0) >= c.c(1)
    {
        -100
    } else {
        0
    }
}

fn cdlinvertedhammer(c: &Cx) -> i32 {
    if c.rb(0) < c.avg(BS, 0)
        && c.us(0) > c.avg(SL, 0)
        && c.ls(0) < c.avg(SVS, 0)
        && c.rb_gap_dn(0, 1)
    {
        100
    } else {
        0
    }
}

fn kicking(c: &Cx) -> bool {
    c.col(1) == -c.col(0)
        && c.rb(1) > c.avg(BL, 1)
        && c.us(1) < c.avg(SVS, 1)
        && c.ls(1) < c.avg(SVS, 1)
        && c.rb(0) > c.avg(BL, 0)
        && c.us(0) < c.avg(SVS, 0)
        && c.ls(0) < c.avg(SVS, 0)
        && ((c.col(1) == -1 && c.gap_up(0, 1)) || (c.col(1) == 1 && c.gap_dn(0, 1)))
}

fn cdlkicking(c: &Cx) -> i32 {
    if kicking(c) {
        c.col(0) * 100
    } else {
        0
    }
}

/// Kicking, signed by the colour of the longer of the two marubozu.
fn cdlkickingbylength(c: &Cx) -> i32 {
    if kicking(c) {
        (if c.rb(0) > c.rb(1) { c.col(0) } else { c.col(1) }) * 100
    } else {
        0
    }
}

fn cdlladderbottom(c: &Cx) -> i32 {
    if c.col(4) == -1
        && c.col(3) == -1
        && c.col(2) == -1
        && c.o(4) > c.o(3)
        && c.o(3) > c.o(2)
        && c.c(4) > c.c(3)
        && c.c(3) > c.c(2)
        && c.col(1) == -1
        && c.us(1) > c.avg(SVS, 1)
        && c.col(0) == 1
        && c.o(0) > c.o(1)
        && c.c(0) > c.h(1)
    {
        100
    } else {
        0
    }
}

fn cdllongleggeddoji(c: &Cx) -> i32 {
    if c.rb(0) <= c.avg(BD, 0) && (c.ls(0) > c.avg(SL, 0) || c.us(0) > c.avg(SL, 0)) {
        100
    } else {
        0
    }
}

fn cdllongline(c: &Cx) -> i32 {
    if c.rb(0) > c.avg(BL, 0) && c.us(0) < c.avg(SS, 0) && c.ls(0) < c.avg(SS, 0) {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdlmarubozu(c: &Cx) -> i32 {
    if c.rb(0) > c.avg(BL, 0) && c.us(0) < c.avg(SVS, 0) && c.ls(0) < c.avg(SVS, 0) {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdlmatchinglow(c: &Cx) -> i32 {
    if c.col(1) == -1
        && c.col(0) == -1
        && c.c(0) <= c.c(1) + c.avg(EQ, 1)
        && c.c(0) >= c.c(1) - c.avg(EQ, 1)
    {
        100
    } else {
        0
    }
}

fn cdlmathold(c: &Cx) -> i32 {
    if c.rb(4) > c.avg(BL, 4)
        && c.rb(3) < c.avg(BS, 3)
        && c.rb(2) < c.avg(BS, 2)
        && c.rb(1) < c.avg(BS, 1)
        && c.col(4) == 1
        && c.col(3) == -1
        && c.col(0) == 1
        && c.rb_gap_up(3, 4)
        && c.bot(2) < c.c(4)
        && c.bot(1) < c.c(4)
        && c.bot(2) > c.c(4) - c.rb(4) * c.pen
        && c.bot(1) > c.c(4) - c.rb(4) * c.pen
        && c.top(2) < c.o(3)
        && c.top(1) < c.top(2)
        && c.o(0) > c.c(1)
        && c.c(0) > c.h(3).max(c.h(2)).max(c.h(1))
    {
        100
    } else {
        0
    }
}

fn cdlmorningdojistar(c: &Cx) -> i32 {
    if c.rb(2) > c.avg(BL, 2)
        && c.col(2) == -1
        && c.rb(1) <= c.avg(BD, 1)
        && c.rb_gap_dn(1, 2)
        && c.rb(0) > c.avg(BS, 0)
        && c.col(0) == 1
        && c.c(0) > c.c(2) + c.rb(2) * c.pen
    {
        100
    } else {
        0
    }
}

fn cdlmorningstar(c: &Cx) -> i32 {
    if c.rb(2) > c.avg(BL, 2)
        && c.col(2) == -1
        && c.rb(1) <= c.avg(BS, 1)
        && c.rb_gap_dn(1, 2)
        && c.rb(0) > c.avg(BS, 0)
        && c.col(0) == 1
        && c.c(0) > c.c(2) + c.rb(2) * c.pen
    {
        100
    } else {
        0
    }
}

fn cdlonneck(c: &Cx) -> i32 {
    if c.col(1) == -1
        && c.rb(1) > c.avg(BL, 1)
        && c.col(0) == 1
        && c.o(0) < c.l(1)
        && c.c(0) <= c.l(1) + c.avg(EQ, 1)
        && c.c(0) >= c.l(1) - c.avg(EQ, 1)
    {
        -100
    } else {
        0
    }
}

fn cdlpiercing(c: &Cx) -> i32 {
    if c.col(1) == -1
        && c.rb(1) > c.avg(BL, 1)
        && c.col(0) == 1
        && c.rb(0) > c.avg(BL, 0)
        && c.o(0) < c.l(1)
        && c.c(0) < c.o(1)
        && c.c(0) > c.c(1) + c.rb(1) * 0.5
    {
        100
    } else {
        0
    }
}

fn cdlrickshawman(c: &Cx) -> i32 {
    if c.rb(0) <= c.avg(BD, 0)
        && c.ls(0) > c.avg(SL, 0)
        && c.us(0) > c.avg(SL, 0)
        && c.bot(0) <= c.l(0) + c.hl(0) / 2.0 + c.avg(NR, 0)
        && c.top(0) >= c.l(0) + c.hl(0) / 2.0 - c.avg(NR, 0)
    {
        100
    } else {
        0
    }
}

fn cdlrisefall3methods(c: &Cx) -> i32 {
    let k = c.col(4);
    let kf = f64::from(k);
    if c.rb(4) > c.avg(BL, 4)
        && c.rb(3) < c.avg(BS, 3)
        && c.rb(2) < c.avg(BS, 2)
        && c.rb(1) < c.avg(BS, 1)
        && c.rb(0) > c.avg(BL, 0)
        && c.col(3) == -k
        && c.col(2) == -k
        && c.col(1) == -k
        && c.col(0) == k
        && c.bot(3) < c.h(4)
        && c.top(3) > c.l(4)
        && c.bot(2) < c.h(4)
        && c.top(2) > c.l(4)
        && c.bot(1) < c.h(4)
        && c.top(1) > c.l(4)
        && c.c(2) * kf < c.c(3) * kf
        && c.c(1) * kf < c.c(2) * kf
        && c.o(0) * kf > c.c(1) * kf
        && c.c(0) * kf > c.c(4) * kf
    {
        100 * k
    } else {
        0
    }
}

fn cdlseparatinglines(c: &Cx) -> i32 {
    if c.col(1) == -c.col(0)
        && c.o(0) <= c.o(1) + c.avg(EQ, 1)
        && c.o(0) >= c.o(1) - c.avg(EQ, 1)
        && c.rb(0) > c.avg(BL, 0)
        && ((c.col(0) == 1 && c.ls(0) < c.avg(SVS, 0))
            || (c.col(0) == -1 && c.us(0) < c.avg(SVS, 0)))
    {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdlshootingstar(c: &Cx) -> i32 {
    if c.rb(0) < c.avg(BS, 0)
        && c.us(0) > c.avg(SL, 0)
        && c.ls(0) < c.avg(SVS, 0)
        && c.rb_gap_up(0, 1)
    {
        -100
    } else {
        0
    }
}

fn cdlshortline(c: &Cx) -> i32 {
    if c.rb(0) < c.avg(BS, 0) && c.us(0) < c.avg(SS, 0) && c.ls(0) < c.avg(SS, 0) {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdlspinningtop(c: &Cx) -> i32 {
    if c.rb(0) < c.avg(BS, 0) && c.us(0) > c.rb(0) && c.ls(0) > c.rb(0) {
        c.col(0) * 100
    } else {
        0
    }
}

fn cdlstalledpattern(c: &Cx) -> i32 {
    if c.col(2) == 1
        && c.col(1) == 1
        && c.col(0) == 1
        && c.c(0) > c.c(1)
        && c.c(1) > c.c(2)
        && c.rb(2) > c.avg(BL, 2)
        && c.rb(1) > c.avg(BL, 1)
        && c.us(1) < c.avg(SVS, 1)
        && c.o(1) > c.o(2)
        && c.o(1) <= c.c(2) + c.avg(NR, 2)
        && c.rb(0) < c.avg(BS, 0)
        && c.o(0) >= c.c(1) - c.rb(0) - c.avg(NR, 1)
    {
        -100
    } else {
        0
    }
}

fn cdlsticksandwich(c: &Cx) -> i32 {
    if c.col(2) == -1
        && c.col(1) == 1
        && c.col(0) == -1
        && c.l(1) > c.c(2)
        && c.c(0) <= c.c(2) + c.avg(EQ, 2)
        && c.c(0) >= c.c(2) - c.avg(EQ, 2)
    {
        100
    } else {
        0
    }
}

fn cdltakuri(c: &Cx) -> i32 {
    if c.rb(0) <= c.avg(BD, 0) && c.us(0) < c.avg(SVS, 0) && c.ls(0) > c.avg(SVL, 0) {
        100
    } else {
        0
    }
}

fn cdltasukigap(c: &Cx) -> i32 {
    if c.rb_gap_up(1, 2)
        && c.col(1) == 1
        && c.col(0) == -1
        && c.o(0) < c.c(1)
        && c.o(0) > c.o(1)
        && c.c(0) < c.o(1)
        && c.c(0) > c.top(2)
        && (c.rb(1) - c.rb(0)).abs() < c.avg(NR, 1)
    {
        return 100;
    }
    if c.rb_gap_dn(1, 2)
        && c.col(1) == -1
        && c.col(0) == 1
        && c.o(0) > c.c(1)
        && c.o(0) < c.o(1)
        && c.c(0) > c.o(1)
        && c.c(0) < c.bot(2)
        && (c.rb(1) - c.rb(0)).abs() < c.avg(NR, 1)
    {
        return -100;
    }
    0
}

fn cdlthrusting(c: &Cx) -> i32 {
    if c.col(1) == -1
        && c.rb(1) > c.avg(BL, 1)
        && c.col(0) == 1
        && c.o(0) < c.l(1)
        && c.c(0) > c.c(1) + c.avg(EQ, 1)
        && c.c(0) <= c.c(1) + c.rb(1) * 0.5
    {
        -100
    } else {
        0
    }
}

fn cdltristar(c: &Cx) -> i32 {
    // All three bodies are compared with the average at the first bar.
    if c.rb(2) <= c.avg(BD, 2) && c.rb(1) <= c.avg(BD, 2) && c.rb(0) <= c.avg(BD, 2) {
        let mut r = 0;
        if c.rb_gap_up(1, 2) && c.top(0) < c.top(1) {
            r = -100;
        }
        if c.rb_gap_dn(1, 2) && c.bot(0) > c.bot(1) {
            r = 100;
        }
        return r;
    }
    0
}

fn cdlunique3river(c: &Cx) -> i32 {
    if c.rb(2) > c.avg(BL, 2)
        && c.col(2) == -1
        && c.col(1) == -1
        && c.c(1) > c.c(2)
        && c.o(1) <= c.o(2)
        && c.l(1) < c.l(2)
        && c.rb(0) < c.avg(BS, 0)
        && c.col(0) == 1
        && c.o(0) > c.l(1)
    {
        100
    } else {
        0
    }
}

fn cdlupsidegap2crows(c: &Cx) -> i32 {
    if c.col(2) == 1
        && c.rb(2) > c.avg(BL, 2)
        && c.col(1) == -1
        && c.rb(1) <= c.avg(BS, 1)
        && c.rb_gap_up(1, 2)
        && c.col(0) == -1
        && c.o(0) > c.o(1)
        && c.c(0) < c.c(1)
        && c.c(0) > c.c(2)
    {
        -100
    } else {
        0
    }
}

fn cdlxsidegap3methods(c: &Cx) -> i32 {
    if c.col(2) == c.col(1) && c.col(1) == -c.col(0) {
        if c.rb_gap_up(1, 2)
            && c.col(2) == 1
            && c.o(0) < c.c(1)
            && c.o(0) > c.o(1)
            && c.c(0) < c.o(1)
            && c.c(0) > c.o(2)
            && c.c(0) < c.c(2)
        {
            return 100;
        }
        if c.rb_gap_dn(1, 2)
            && c.col(2) == -1
            && c.o(0) > c.c(1)
            && c.o(0) < c.o(1)
            && c.c(0) > c.o(1)
            && c.c(0) < c.o(2)
            && c.c(0) > c.c(2)
        {
            return -100;
        }
    }
    0
}

// ------------------------------------------------------------------------------ kernel

/// What a pattern needs: the bars its shape spans, the averages it reads and its function.
pub struct Spec {
    /// The expression name, TA-Lib's lower-cased.
    name: &'static str,
    /// Bars the shape needs before the current one.
    bars: usize,
    /// Floor on the averaging period in the lookback (hikkakemod: `max(1, Near)`).
    min_period: usize,
    /// Every (setting, bar offset) whose running average the pattern reads.
    avgs: &'static [(Set, usize)],
    /// `penetration`, TA-Lib's per-pattern ratio, on the patterns that have one.
    params: &'static [ParamDef],
    eval: fn(&Cx) -> i32,
}

impl Spec {
    /// TA-Lib's lookback under `settings`: the shape's bars plus the longest averaging
    /// period among the settings read.
    pub fn lookback(&self, settings: &CandleSettings) -> usize {
        let p = self.avgs.iter().map(|&(set, _)| settings.get(set).period).max().unwrap_or(0);
        self.bars + p.max(self.min_period)
    }
}

struct CandleKernel {
    spec: &'static Spec,
    set: CandleSettings,
    pen: f64,
    lb: usize,
    /// The last `lb + 2` bars, newest first, in a buffer of twice that length with every bar
    /// written at `pos` and `pos + lb + 2`: `buf[pos..pos + lb + 2]` is always contiguous with
    /// bar `j` back at index `j` (one more than the window, which the oldest range leaving a
    /// running total comes from), and a step writes two bars instead of copying the window.
    buf: Vec<[f64; 4]>,
    pos: usize,
    /// Bars among the last `lb + 1` that hold a non-finite value; a bar that does not exist
    /// yet is NaN, so the first `lb` bars end in NaN too.
    bad: usize,
    tot: Vec<f64>,
    slot: [[u8; MAX_OFF]; N_SETS],
    /// The previous bar had a clean window, so the totals are current.
    live: bool,
}

impl CandleKernel {
    fn new(spec: &'static Spec, set: CandleSettings, pen: f64) -> Self {
        let lb = spec.lookback(&set);
        let mut slot = [[NO_AVG; MAX_OFF]; N_SETS];
        for (t, &(s, off)) in spec.avgs.iter().enumerate() {
            slot[s as usize][off] = u8::try_from(t).expect("few averages per pattern");
        }
        Self {
            spec,
            set,
            pen,
            lb,
            buf: vec![[f64::NAN; 4]; 2 * (lb + 2)],
            pos: 0,
            bad: lb + 1,
            tot: vec![0.0; spec.avgs.len()],
            slot,
            live: false,
        }
    }
}

fn is_clean(b: &[f64; 4]) -> bool {
    b.iter().all(|v| v.is_finite())
}

impl Kernel<4, 1> for CandleKernel {
    fn heap_bytes(&self) -> usize {
        self.buf.capacity() * std::mem::size_of::<[f64; 4]>()
            + self.tot.capacity() * std::mem::size_of::<f64>()
    }

    /// Bars are held here, not read from the history.
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<4>>(&mut self, x: [f64; 4], _h: &H) -> [f64; 1] {
        let len = self.lb + 2;
        if !is_clean(&self.buf[self.pos + self.lb]) {
            self.bad -= 1;
        }
        if !is_clean(&x) {
            self.bad += 1;
        }
        self.pos = if self.pos == 0 { len - 1 } else { self.pos - 1 };
        self.buf[self.pos] = x;
        self.buf[self.pos + len] = x;
        if self.bad > 0 {
            self.live = false;
            return [f64::NAN];
        }
        let bars = &self.buf[self.pos..self.pos + len];
        for (t, &(set, off)) in self.spec.avgs.iter().enumerate() {
            let s = self.set.get(set);
            if s.period == 0 {
                continue;
            }
            if self.live {
                let add = range_of(bars, s.range, off + 1);
                let drop = range_of(bars, s.range, off + 1 + s.period);
                self.tot[t] += add - drop;
            } else {
                let mut sum = 0.0;
                for k in (off + 1..=off + s.period).rev() {
                    sum += range_of(bars, s.range, k);
                }
                self.tot[t] = sum;
            }
        }
        self.live = true;
        let cx = Cx { b: bars, set: &self.set, slot: &self.slot, tot: &self.tot, pen: self.pen };
        [f64::from((self.spec.eval)(&cx))]
    }
}

fn penetration(spec: &Spec, params: &[ParamVal]) -> Result<f64, String> {
    if spec.params.is_empty() {
        return Ok(0.0);
    }
    match params.first() {
        Some(ParamVal::Float(v)) if v.is_finite() && *v >= 0.0 => Ok(*v),
        Some(ParamVal::Int(v)) if *v >= 0 => Ok(*v as f64),
        other => {
            Err(format!("{}: penetration must be a non-negative number, got {other:?}", spec.name))
        }
    }
}

fn make_node(
    spec: &'static Spec,
    set: CandleSettings,
    params: &[ParamVal],
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let k = CandleKernel::new(spec, set, penetration(spec, params)?);
    Ok(Box::new(Adapter::new(
        k,
        regs::<4>(ins, "candle inputs")?,
        regs::<1>(outs, "candle outputs")?,
    )))
}

/// A pattern node on a settings table of the caller's. Expressions never take settings; this
/// is how Rust code (the changed-settings replay) builds a node with its own. `name` is the
/// expression name (`cdlhammer`).
pub fn make_with_settings(
    name: &str,
    settings: CandleSettings,
    params: &[ParamVal],
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let spec = SPECS
        .iter()
        .copied()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("no candlestick pattern named {name}"))?;
    make_node(spec, settings, params, ins, outs)
}

/// The warm-up of the named pattern's node under `settings`.
pub fn lookback_with_settings(name: &str, settings: &CandleSettings) -> Option<usize> {
    SPECS.iter().find(|s| s.name == name).map(|s| s.lookback(settings))
}

const OHLC: &[InputDef] = &[
    InputDef { name: "open", default_field: Some("open") },
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
];

const NO_PARAMS: &[ParamDef] = &[];
const PEN_03: &[ParamDef] = &[ParamDef {
    name: "penetration",
    kind: ParamKind::Float { min: 0.0, max: 1.0e6 },
    default: Some(ParamVal::Float(0.3)),
}];
const PEN_05: &[ParamDef] = &[ParamDef {
    name: "penetration",
    kind: ParamKind::Float { min: 0.0, max: 1.0e6 },
    default: Some(ParamVal::Float(0.5)),
}];

/// One row per pattern: spec name, TA-Lib name, bars, lookback floor, averages read, params,
/// function, doc.
macro_rules! candles {
    ($( $id:ident { $name:literal, $talib:literal, $bars:expr, $min:expr, $avgs:expr, $params:expr, $eval:ident, $doc:expr } )*) => {
        $(
            const $id: Spec = Spec {
                name: $name,
                bars: $bars,
                min_period: $min,
                avgs: $avgs,
                params: $params,
                eval: $eval,
            };
        )*

        static SPECS: &[&Spec] = &[ $( &$id ),* ];

        pub const DEFS: &[FnDef] = &[
            $(
                FnDef {
                    name: $name,
                    family: Family::Candles,
                    inputs: OHLC,
                    params: $params,
                    outputs: &["value"],
                    out_kind: OutKind::Series,
                    warmup: |_| $id.lookback(&CandleSettings::TALIB),
                    memory: Memory::Finite,
                    legacy: None,
                    talib_name: Some($talib),
                    doc: $doc,
                    refs: &[],
                    rev: 1,
                    invariance: &[Invariance::ScaleInvariant],
                    causal: true,
                    nan_rule: NanRule::Windowed,
                    domain: Domain::Any,
                    reference: Reference::Fixture,
                    relation: Relation::Candle,
                    make: |p, i, o| make_node(&$id, CandleSettings::TALIB, p, i, o),
                },
            )*
        ];
    };
}

candles! {
    CDL2CROWS { "cdl2crows", "CDL2CROWS", 2, 0, &[(BL, 2)], NO_PARAMS, cdl2crows, "Two Crows (TA-Lib CDL2CROWS): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDL3BLACKCROWS { "cdl3blackcrows", "CDL3BLACKCROWS", 3, 0, &[(SVS, 2), (SVS, 1), (SVS, 0)], NO_PARAMS, cdl3blackcrows, "Three Black Crows (TA-Lib CDL3BLACKCROWS): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDL3INSIDE { "cdl3inside", "CDL3INSIDE", 2, 0, &[(BL, 2), (BS, 1)], NO_PARAMS, cdl3inside, "Three Inside Up/Down (TA-Lib CDL3INSIDE): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDL3LINESTRIKE { "cdl3linestrike", "CDL3LINESTRIKE", 3, 0, &[(NR, 3), (NR, 2)], NO_PARAMS, cdl3linestrike, "Three-Line Strike  (TA-Lib CDL3LINESTRIKE): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDL3OUTSIDE { "cdl3outside", "CDL3OUTSIDE", 3, 0, &[], NO_PARAMS, cdl3outside, "Three Outside Up/Down (TA-Lib CDL3OUTSIDE): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDL3STARSINSOUTH { "cdl3starsinsouth", "CDL3STARSINSOUTH", 2, 0, &[(SVS, 1), (SVS, 0), (SL, 2), (SL, 1), (BL, 2), (BS, 0)], NO_PARAMS, cdl3starsinsouth, "Three Stars In The South (TA-Lib CDL3STARSINSOUTH): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDL3WHITESOLDIERS { "cdl3whitesoldiers", "CDL3WHITESOLDIERS", 2, 0, &[(SVS, 2), (SVS, 1), (SVS, 0), (NR, 2), (NR, 1), (FR, 2), (FR, 1), (BS, 0)], NO_PARAMS, cdl3whitesoldiers, "Three Advancing White Soldiers (TA-Lib CDL3WHITESOLDIERS): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLABANDONEDBABY { "cdlabandonedbaby", "CDLABANDONEDBABY", 2, 0, &[(BD, 1), (BL, 2), (BS, 0)], PEN_03, cdlabandonedbaby, "Abandoned Baby (TA-Lib CDLABANDONEDBABY): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. `penetration` is the required depth of the third candle into the first body (default 0.3)." }
    CDLADVANCEBLOCK { "cdladvanceblock", "CDLADVANCEBLOCK", 2, 0, &[(SS, 2), (SS, 1), (SS, 0), (SL, 0), (NR, 2), (NR, 1), (FR, 2), (FR, 1), (BL, 2)], NO_PARAMS, cdladvanceblock, "Advance Block (TA-Lib CDLADVANCEBLOCK): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLBELTHOLD { "cdlbelthold", "CDLBELTHOLD", 0, 0, &[(BL, 0), (SVS, 0)], NO_PARAMS, cdlbelthold, "Belt-hold (TA-Lib CDLBELTHOLD): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLBREAKAWAY { "cdlbreakaway", "CDLBREAKAWAY", 4, 0, &[(BL, 4)], NO_PARAMS, cdlbreakaway, "Breakaway (TA-Lib CDLBREAKAWAY): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLCLOSINGMARUBOZU { "cdlclosingmarubozu", "CDLCLOSINGMARUBOZU", 0, 0, &[(BL, 0), (SVS, 0)], NO_PARAMS, cdlclosingmarubozu, "Closing Marubozu (TA-Lib CDLCLOSINGMARUBOZU): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLCONCEALBABYSWALL { "cdlconcealbabyswall", "CDLCONCEALBABYSWALL", 3, 0, &[(SVS, 3), (SVS, 2), (SVS, 1)], NO_PARAMS, cdlconcealbabyswall, "Concealing Baby Swallow (TA-Lib CDLCONCEALBABYSWALL): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLCOUNTERATTACK { "cdlcounterattack", "CDLCOUNTERATTACK", 1, 0, &[(EQ, 1), (BL, 1), (BL, 0)], NO_PARAMS, cdlcounterattack, "Counterattack (TA-Lib CDLCOUNTERATTACK): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLDARKCLOUDCOVER { "cdldarkcloudcover", "CDLDARKCLOUDCOVER", 1, 0, &[(BL, 1)], PEN_05, cdldarkcloudcover, "Dark Cloud Cover (TA-Lib CDLDARKCLOUDCOVER): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. `penetration` is the required depth of the third candle into the first body (default 0.5)." }
    CDLDOJI { "cdldoji", "CDLDOJI", 0, 0, &[(BD, 0)], NO_PARAMS, cdldoji, "Doji (TA-Lib CDLDOJI): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLDOJISTAR { "cdldojistar", "CDLDOJISTAR", 1, 0, &[(BL, 1), (BD, 0)], NO_PARAMS, cdldojistar, "Doji Star (TA-Lib CDLDOJISTAR): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLDRAGONFLYDOJI { "cdldragonflydoji", "CDLDRAGONFLYDOJI", 0, 0, &[(BD, 0), (SVS, 0)], NO_PARAMS, cdldragonflydoji, "Dragonfly Doji (TA-Lib CDLDRAGONFLYDOJI): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLENGULFING { "cdlengulfing", "CDLENGULFING", 2, 0, &[], NO_PARAMS, cdlengulfing, "Engulfing Pattern (TA-Lib CDLENGULFING): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. +-80 when the body ties the previous one on one side." }
    CDLEVENINGDOJISTAR { "cdleveningdojistar", "CDLEVENINGDOJISTAR", 2, 0, &[(BL, 2), (BD, 1), (BS, 0)], PEN_03, cdleveningdojistar, "Evening Doji Star (TA-Lib CDLEVENINGDOJISTAR): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. `penetration` is the required depth of the third candle into the first body (default 0.3)." }
    CDLEVENINGSTAR { "cdleveningstar", "CDLEVENINGSTAR", 2, 0, &[(BL, 2), (BS, 1), (BS, 0)], PEN_03, cdleveningstar, "Evening Star (TA-Lib CDLEVENINGSTAR): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. `penetration` is the required depth of the third candle into the first body (default 0.3)." }
    CDLGAPSIDESIDEWHITE { "cdlgapsidesidewhite", "CDLGAPSIDESIDEWHITE", 2, 0, &[(NR, 1), (EQ, 1)], NO_PARAMS, cdlgapsidesidewhite, "Up/Down-gap side-by-side white lines (TA-Lib CDLGAPSIDESIDEWHITE): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLGRAVESTONEDOJI { "cdlgravestonedoji", "CDLGRAVESTONEDOJI", 0, 0, &[(BD, 0), (SVS, 0)], NO_PARAMS, cdlgravestonedoji, "Gravestone Doji (TA-Lib CDLGRAVESTONEDOJI): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLHAMMER { "cdlhammer", "CDLHAMMER", 1, 0, &[(BS, 0), (SL, 0), (SVS, 0), (NR, 1)], NO_PARAMS, cdlhammer, "Hammer (TA-Lib CDLHAMMER): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLHANGINGMAN { "cdlhangingman", "CDLHANGINGMAN", 1, 0, &[(BS, 0), (SL, 0), (SVS, 0), (NR, 1)], NO_PARAMS, cdlhangingman, "Hanging Man (TA-Lib CDLHANGINGMAN): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLHARAMI { "cdlharami", "CDLHARAMI", 1, 0, &[(BL, 1), (BS, 0)], NO_PARAMS, cdlharami, "Harami Pattern (TA-Lib CDLHARAMI): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. +-80 when the small body touches the long one." }
    CDLHARAMICROSS { "cdlharamicross", "CDLHARAMICROSS", 1, 0, &[(BL, 1), (BD, 0)], NO_PARAMS, cdlharamicross, "Harami Cross Pattern (TA-Lib CDLHARAMICROSS): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. +-80 when the doji touches the long body." }
    CDLHIGHWAVE { "cdlhighwave", "CDLHIGHWAVE", 0, 0, &[(BS, 0), (SVL, 0)], NO_PARAMS, cdlhighwave, "High-Wave Candle (TA-Lib CDLHIGHWAVE): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLHIKKAKE { "cdlhikkake", "CDLHIKKAKE", 5, 0, &[], NO_PARAMS, cdlhikkake, "Hikkake Pattern (TA-Lib CDLHIKKAKE): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. +-200 on the bar that confirms a pattern within three bars." }
    CDLHIKKAKEMOD { "cdlhikkakemod", "CDLHIKKAKEMOD", 5, 1, &[(NR, 5), (NR, 4), (NR, 3), (NR, 2)], NO_PARAMS, cdlhikkakemod, "Modified Hikkake Pattern (TA-Lib CDLHIKKAKEMOD): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. +-200 on the bar that confirms a pattern within three bars." }
    CDLHOMINGPIGEON { "cdlhomingpigeon", "CDLHOMINGPIGEON", 1, 0, &[(BL, 1), (BS, 0)], NO_PARAMS, cdlhomingpigeon, "Homing Pigeon (TA-Lib CDLHOMINGPIGEON): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLIDENTICAL3CROWS { "cdlidentical3crows", "CDLIDENTICAL3CROWS", 2, 0, &[(SVS, 2), (SVS, 1), (SVS, 0), (EQ, 2), (EQ, 1)], NO_PARAMS, cdlidentical3crows, "Identical Three Crows (TA-Lib CDLIDENTICAL3CROWS): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLINNECK { "cdlinneck", "CDLINNECK", 1, 0, &[(EQ, 1), (BL, 1)], NO_PARAMS, cdlinneck, "In-Neck Pattern (TA-Lib CDLINNECK): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLINVERTEDHAMMER { "cdlinvertedhammer", "CDLINVERTEDHAMMER", 1, 0, &[(BS, 0), (SL, 0), (SVS, 0)], NO_PARAMS, cdlinvertedhammer, "Inverted Hammer (TA-Lib CDLINVERTEDHAMMER): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLKICKING { "cdlkicking", "CDLKICKING", 1, 0, &[(SVS, 1), (SVS, 0), (BL, 1), (BL, 0)], NO_PARAMS, cdlkicking, "Kicking (TA-Lib CDLKICKING): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLKICKINGBYLENGTH { "cdlkickingbylength", "CDLKICKINGBYLENGTH", 1, 0, &[(SVS, 1), (SVS, 0), (BL, 1), (BL, 0)], NO_PARAMS, cdlkickingbylength, "Kicking - bull/bear determined by the longer marubozu (TA-Lib CDLKICKINGBYLENGTH): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLLADDERBOTTOM { "cdlladderbottom", "CDLLADDERBOTTOM", 4, 0, &[(SVS, 1)], NO_PARAMS, cdlladderbottom, "Ladder Bottom (TA-Lib CDLLADDERBOTTOM): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLLONGLEGGEDDOJI { "cdllongleggeddoji", "CDLLONGLEGGEDDOJI", 0, 0, &[(BD, 0), (SL, 0)], NO_PARAMS, cdllongleggeddoji, "Long Legged Doji (TA-Lib CDLLONGLEGGEDDOJI): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLLONGLINE { "cdllongline", "CDLLONGLINE", 0, 0, &[(BL, 0), (SS, 0)], NO_PARAMS, cdllongline, "Long Line Candle (TA-Lib CDLLONGLINE): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLMARUBOZU { "cdlmarubozu", "CDLMARUBOZU", 0, 0, &[(BL, 0), (SVS, 0)], NO_PARAMS, cdlmarubozu, "Marubozu (TA-Lib CDLMARUBOZU): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLMATCHINGLOW { "cdlmatchinglow", "CDLMATCHINGLOW", 1, 0, &[(EQ, 1)], NO_PARAMS, cdlmatchinglow, "Matching Low (TA-Lib CDLMATCHINGLOW): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLMATHOLD { "cdlmathold", "CDLMATHOLD", 4, 0, &[(BL, 4), (BS, 3), (BS, 2), (BS, 1)], PEN_05, cdlmathold, "Mat Hold (TA-Lib CDLMATHOLD): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. `penetration` is the required depth of the third candle into the first body (default 0.5)." }
    CDLMORNINGDOJISTAR { "cdlmorningdojistar", "CDLMORNINGDOJISTAR", 2, 0, &[(BL, 2), (BD, 1), (BS, 0)], PEN_03, cdlmorningdojistar, "Morning Doji Star (TA-Lib CDLMORNINGDOJISTAR): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. `penetration` is the required depth of the third candle into the first body (default 0.3)." }
    CDLMORNINGSTAR { "cdlmorningstar", "CDLMORNINGSTAR", 2, 0, &[(BL, 2), (BS, 1), (BS, 0)], PEN_03, cdlmorningstar, "Morning Star (TA-Lib CDLMORNINGSTAR): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent. `penetration` is the required depth of the third candle into the first body (default 0.3)." }
    CDLONNECK { "cdlonneck", "CDLONNECK", 1, 0, &[(EQ, 1), (BL, 1)], NO_PARAMS, cdlonneck, "On-Neck Pattern (TA-Lib CDLONNECK): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLPIERCING { "cdlpiercing", "CDLPIERCING", 1, 0, &[(BL, 1), (BL, 0)], NO_PARAMS, cdlpiercing, "Piercing Pattern (TA-Lib CDLPIERCING): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLRICKSHAWMAN { "cdlrickshawman", "CDLRICKSHAWMAN", 0, 0, &[(BD, 0), (SL, 0), (NR, 0)], NO_PARAMS, cdlrickshawman, "Rickshaw Man (TA-Lib CDLRICKSHAWMAN): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLRISEFALL3METHODS { "cdlrisefall3methods", "CDLRISEFALL3METHODS", 4, 0, &[(BL, 4), (BS, 3), (BS, 2), (BS, 1), (BL, 0)], NO_PARAMS, cdlrisefall3methods, "Rising/Falling Three Methods (TA-Lib CDLRISEFALL3METHODS): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLSEPARATINGLINES { "cdlseparatinglines", "CDLSEPARATINGLINES", 1, 0, &[(SVS, 0), (BL, 0), (EQ, 1)], NO_PARAMS, cdlseparatinglines, "Separating Lines (TA-Lib CDLSEPARATINGLINES): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLSHOOTINGSTAR { "cdlshootingstar", "CDLSHOOTINGSTAR", 1, 0, &[(BS, 0), (SL, 0), (SVS, 0)], NO_PARAMS, cdlshootingstar, "Shooting Star (TA-Lib CDLSHOOTINGSTAR): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLSHORTLINE { "cdlshortline", "CDLSHORTLINE", 0, 0, &[(BS, 0), (SS, 0)], NO_PARAMS, cdlshortline, "Short Line Candle (TA-Lib CDLSHORTLINE): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLSPINNINGTOP { "cdlspinningtop", "CDLSPINNINGTOP", 0, 0, &[(BS, 0)], NO_PARAMS, cdlspinningtop, "Spinning Top (TA-Lib CDLSPINNINGTOP): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLSTALLEDPATTERN { "cdlstalledpattern", "CDLSTALLEDPATTERN", 2, 0, &[(BL, 2), (BL, 1), (SVS, 1), (NR, 2), (NR, 1), (BS, 0)], NO_PARAMS, cdlstalledpattern, "Stalled Pattern (TA-Lib CDLSTALLEDPATTERN): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLSTICKSANDWICH { "cdlsticksandwich", "CDLSTICKSANDWICH", 2, 0, &[(EQ, 2)], NO_PARAMS, cdlsticksandwich, "Stick Sandwich (TA-Lib CDLSTICKSANDWICH): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLTAKURI { "cdltakuri", "CDLTAKURI", 0, 0, &[(BD, 0), (SVS, 0), (SVL, 0)], NO_PARAMS, cdltakuri, "Takuri (Dragonfly Doji with very long lower shadow) (TA-Lib CDLTAKURI): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLTASUKIGAP { "cdltasukigap", "CDLTASUKIGAP", 2, 0, &[(NR, 1)], NO_PARAMS, cdltasukigap, "Tasuki Gap (TA-Lib CDLTASUKIGAP): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLTHRUSTING { "cdlthrusting", "CDLTHRUSTING", 1, 0, &[(EQ, 1), (BL, 1)], NO_PARAMS, cdlthrusting, "Thrusting Pattern (TA-Lib CDLTHRUSTING): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLTRISTAR { "cdltristar", "CDLTRISTAR", 2, 0, &[(BD, 2)], NO_PARAMS, cdltristar, "Tristar Pattern (TA-Lib CDLTRISTAR): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLUNIQUE3RIVER { "cdlunique3river", "CDLUNIQUE3RIVER", 2, 0, &[(BL, 2), (BS, 0)], NO_PARAMS, cdlunique3river, "Unique 3 River (TA-Lib CDLUNIQUE3RIVER): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLUPSIDEGAP2CROWS { "cdlupsidegap2crows", "CDLUPSIDEGAP2CROWS", 2, 0, &[(BL, 2), (BS, 1)], NO_PARAMS, cdlupsidegap2crows, "Upside Gap Two Crows (TA-Lib CDLUPSIDEGAP2CROWS): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
    CDLXSIDEGAP3METHODS { "cdlxsidegap3methods", "CDLXSIDEGAP3METHODS", 2, 0, &[], NO_PARAMS, cdlxsidegap3methods, "Upside/Downside Gap Three Methods (TA-Lib CDLXSIDEGAP3METHODS): +100 for a bullish signal, -100 for a bearish one, 0 when the pattern is absent." }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::kernel::{canon_bits, Regs};
    use crate::quant::registry::{all, lookup};
    use serde_json::Value;

    fn defaults(name: &str) -> Vec<ParamVal> {
        lookup(name).unwrap().params.iter().map(|p| p.default.unwrap()).collect()
    }

    fn pattern_names() -> Vec<&'static str> {
        all().filter(|d| d.family == Family::Candles).map(|d| d.name).collect()
    }

    fn run(name: &str, set: CandleSettings, bars: &[[f64; 4]]) -> Vec<f64> {
        run_with(name, set, &defaults(name), bars)
    }

    fn run_with(
        name: &str,
        set: CandleSettings,
        params: &[ParamVal],
        bars: &[[f64; 4]],
    ) -> Vec<f64> {
        let col = |k: usize| -> Vec<f64> { bars.iter().map(|b| b[k]).collect() };
        let (o, h, l, c) = (col(0), col(1), col(2), col(3));
        let mut out = vec![0.0; bars.len()];
        make_with_settings(name, set, params, &[0, 1, 2, 3], &[4])
            .unwrap()
            .run_batch(&[&o, &h, &l, &c], &mut [&mut out[..]]);
        out
    }

    fn stream(name: &str, set: CandleSettings, bars: &[[f64; 4]]) -> Vec<f64> {
        let mut node = make_with_settings(name, set, &defaults(name), &[0, 1, 2, 3], &[4]).unwrap();
        let mut regs = Regs::new(5);
        bars.iter()
            .map(|b| {
                for (k, v) in b.iter().enumerate() {
                    regs.set(k, *v);
                }
                node.step(&mut regs);
                regs.get(4)
            })
            .collect()
    }

    /// Exact equality with NaN equal to NaN; reports the first differing bar.
    fn same_bits(name: &str, what: &str, base: usize, a: &[f64], b: &[f64]) {
        if let Some(i) = (0..a.len()).find(|&i| canon_bits(a[i]) != canon_bits(b[i])) {
            panic!("{name}: {what}: bar {} differs, {} vs {}", base + i, a[i], b[i]);
        }
    }

    fn nan_at(out: &[f64], from: usize, to: usize) -> bool {
        out[from..to].iter().all(|v| v.is_nan())
    }

    /// Ten bars with body 4 and range 4, no shadows: every average over them is easy.
    fn fillers() -> Vec<[f64; 4]> {
        vec![[100.0, 104.0, 100.0, 104.0]; 10]
    }

    #[test]
    fn doji_hand_values() {
        // Ten bars of range 4: the BodyDoji average is 0.1 * (10 * 4) / 10 = 0.4. A body of
        // 0.2 is at most that, so the bar is a doji; the next bar's window drops one filler
        // (range 4) for the doji's range 2, 38 / 10 * 0.1 = 0.38, and a body of 0.5 is not.
        let mut bars = fillers();
        bars.push([100.0, 101.0, 99.0, 100.2]);
        bars.push([100.0, 101.0, 99.0, 100.5]);
        let out = run("cdldoji", CandleSettings::TALIB, &bars);
        assert!(nan_at(&out, 0, 10));
        assert_eq!((out[10], out[11]), (100.0, 0.0));
        assert_eq!((lookup("cdldoji").unwrap().warmup)(&[]), 10);
    }

    #[test]
    fn two_crows_hand_values() {
        // Bars 0..9 have body 4, so the BodyLong average for bar 10 is 4. Bar 10: white,
        // body 6 > 4. Bar 11: black, its body (112..114) wholly above bar 10's (104..110).
        // Bar 12: black, opens 113 inside bar 11's body (112..114), closes 107 inside bar 10's
        // body (104..110): two crows, -100. Closing at 111 is above bar 10's close: no pattern.
        let mut bars = fillers();
        bars.push([104.0, 110.0, 104.0, 110.0]);
        bars.push([114.0, 114.0, 112.0, 112.0]);
        bars.push([113.0, 113.0, 107.0, 107.0]);
        let out = run("cdl2crows", CandleSettings::TALIB, &bars);
        assert!(nan_at(&out, 0, 12));
        assert_eq!(out[12], -100.0);
        bars[12] = [113.0, 113.0, 111.0, 111.0];
        assert_eq!(run("cdl2crows", CandleSettings::TALIB, &bars)[12], 0.0);
    }

    #[test]
    fn engulfing_is_100_strict_and_80_tied() {
        // A black bar (10 -> 8) then a white one closing at 12. Opening at 8 ties the black
        // close, so one side is not strict: 80. Opening at 7 is strictly below: 100.
        let mut bars = vec![[9.0, 9.0, 9.0, 9.0], [10.0, 10.0, 8.0, 8.0], [8.0, 12.0, 8.0, 12.0]];
        assert_eq!(run("cdlengulfing", CandleSettings::TALIB, &bars)[2], 80.0);
        bars[2] = [7.0, 12.0, 7.0, 12.0];
        assert_eq!(run("cdlengulfing", CandleSettings::TALIB, &bars)[2], 100.0);
    }

    #[test]
    fn hikkake_confirms_within_three_bars() {
        // Bar 3 (high 20, low 10) holds bar 4 (19, 11) inside it; bar 5 (18, 9) breaks the
        // inside bar's low and not its high: a bullish hikkake, +100, at bar 5. Bar 6 closes
        // at 21, above bar 4's high of 19, which confirms it: 100 + 100. Closing at 18 does
        // not confirm.
        let f = [15.0, 16.0, 14.0, 15.0];
        let mut bars = vec![
            f,
            f,
            f,
            [15.0, 20.0, 10.0, 15.0],
            [15.0, 19.0, 11.0, 15.0],
            [15.0, 18.0, 9.0, 15.0],
            [15.0, 22.0, 14.0, 21.0],
        ];
        let out = run("cdlhikkake", CandleSettings::TALIB, &bars);
        assert!(nan_at(&out, 0, 5));
        assert_eq!((out[5], out[6]), (100.0, 200.0));
        bars[6] = [15.0, 22.0, 14.0, 18.0];
        assert_eq!(run("cdlhikkake", CandleSettings::TALIB, &bars)[6], 0.0);
    }

    #[test]
    fn morning_star_penetration_is_a_parameter() {
        // BodyLong average 4 at bar 10 (black, body 6). BodyShort average at bar 11 is
        // (9 * 4 + 6) / 10 = 4.2: the star's body 1 is small, and its body (101..102) is wholly
        // below bar 10's (104..110). At bar 12 the average is (8 * 4 + 6 + 1) / 10 = 3.9, so
        // the white bar's body 7 is long. It must close above 104 + 6 * pen: at 0.3 that is
        // 105.8 and it closes at 108, at 0.9 it is 109.4 and it does not.
        let mut bars = fillers();
        bars.push([110.0, 110.0, 104.0, 104.0]);
        bars.push([102.0, 102.0, 101.0, 101.0]);
        bars.push([101.0, 108.0, 101.0, 108.0]);
        let at = |pen: f64| {
            run_with("cdlmorningstar", CandleSettings::TALIB, &[ParamVal::Float(pen)], &bars)[12]
        };
        assert_eq!((at(0.3), at(0.9)), (100.0, 0.0));
        assert_eq!(run("cdlmorningstar", CandleSettings::TALIB, &bars)[12], 100.0);
    }

    #[test]
    fn warmup_depends_on_the_settings() {
        // Doji reads BodyDoji only: a period of 5 gives a lookback of 5. Hikkakemod floors the
        // Near period at 1: with Near at 0 the lookback is 5 + 1.
        let s = CandleSettings::TALIB.with(BD, setting(RangeType::HighLow, 5, 0.2));
        assert_eq!(lookback_with_settings("cdldoji", &s), Some(5));
        assert_eq!(lookback_with_settings("cdldoji", &CandleSettings::TALIB), Some(10));
        let near0 = CandleSettings::TALIB.with(NR, setting(RangeType::HighLow, 0, 0.2));
        assert_eq!(lookback_with_settings("cdlhikkakemod", &near0), Some(6));
        let mut bars = Vec::new();
        for i in 0..20 {
            let v = f64::from(i);
            bars.push([v, v + 2.0, v - 1.0, v + 1.0]);
        }
        let out = run("cdldoji", s, &bars);
        assert_eq!(out.iter().position(|v| !v.is_nan()), Some(5));
    }

    /// Integer-valued bars on a coarse grid, so ties are common and every sum is exact.
    fn grid_bars(n: usize, q: u64, mut state: u64) -> Vec<[f64; 4]> {
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        (0..n)
            .map(|_| {
                let o = next() % q;
                let c = (o + next() % q).saturating_sub(q / 2);
                let h = o.max(c) + next() % (q / 2 + 1);
                let l = o.min(c).saturating_sub(next() % (q / 2 + 1));
                [o as f64, h as f64, l as f64, c as f64]
            })
            .collect()
    }

    #[test]
    fn nan_spoils_its_lookback_window_and_the_totals_are_rebuilt() {
        let clean = grid_bars(400, 10, 7);
        for name in pattern_names() {
            let lb = lookback_with_settings(name, &CandleSettings::TALIB).unwrap();
            for field in 0..4 {
                let mut dirty = clean.clone();
                dirty[200][field] = f64::NAN;
                let a = run(name, CandleSettings::TALIB, &clean);
                let b = run(name, CandleSettings::TALIB, &dirty);
                same_bits(name, "causal", 0, &a[..200], &b[..200]);
                assert!(nan_at(&b, 200, 201 + lb), "{name}: NaN while the bar is in the window");
                same_bits(name, "recovers exactly", 201 + lb, &a[201 + lb..], &b[201 + lb..]);
            }
        }
    }

    #[test]
    fn batch_and_streaming_are_bit_identical() {
        let mut bars = grid_bars(300, 12, 11);
        bars[120][1] = f64::NAN;
        bars[121][3] = f64::INFINITY;
        for name in pattern_names() {
            let a = run(name, CandleSettings::TALIB, &bars);
            let b = stream(name, CandleSettings::TALIB, &bars);
            let (a, b): (Vec<u64>, Vec<u64>) =
                (a.into_iter().map(canon_bits).collect(), b.into_iter().map(canon_bits).collect());
            assert_eq!(a, b, "{name}");
        }
    }

    // ---- TA-Lib replay: tests/oracle/candles_changed_extract.json (see extract_candles_changed.py)

    struct Replay {
        bars: Vec<[f64; 4]>,
        json: Value,
    }

    fn replay() -> Replay {
        let path =
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/candles_changed_extract.json");
        let json: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let col = |k: &str| -> Vec<f64> {
            let hex = json["columns"][k].as_str().unwrap();
            (0..hex.len() / 16)
                .map(|i| {
                    f64::from_bits(u64::from_str_radix(&hex[i * 16..i * 16 + 16], 16).unwrap())
                })
                .collect()
        };
        let (o, h, l, c) = (col("open"), col("high"), col("low"), col("close"));
        let bars = (0..o.len()).map(|i| [o[i], h[i], l[i], c[i]]).collect();
        Replay { bars, json }
    }

    fn table(v: &Value) -> CandleSettings {
        let mut s = CandleSettings::TALIB;
        for (k, v) in v.as_object().unwrap() {
            let set = match k.as_str() {
                "BodyLong" => Set::BodyLong,
                "BodyVeryLong" => Set::BodyVeryLong,
                "BodyShort" => Set::BodyShort,
                "BodyDoji" => Set::BodyDoji,
                "ShadowLong" => Set::ShadowLong,
                "ShadowVeryLong" => Set::ShadowVeryLong,
                "ShadowShort" => Set::ShadowShort,
                "ShadowVeryShort" => Set::ShadowVeryShort,
                "Near" => Set::Near,
                "Far" => Set::Far,
                "Equal" => Set::Equal,
                other => panic!("unknown setting {other}"),
            };
            let range = match v[0].as_str().unwrap() {
                "RealBody" => RangeType::RealBody,
                "HighLow" => RangeType::HighLow,
                "Shadows" => RangeType::Shadows,
                other => panic!("unknown range type {other}"),
            };
            s = s
                .with(set, setting(range, v[1].as_u64().unwrap() as usize, v[2].as_f64().unwrap()));
        }
        s
    }

    /// Replay every pattern on `settings` and compare with TA-Lib's integers: NaN before the
    /// lookback, exactly equal from it on.
    fn replay_all(r: &Replay, settings: CandleSettings, which: &str, lookback_key: &str) {
        let patterns = r.json["patterns"].as_object().unwrap();
        assert_eq!(patterns.len(), 61);
        let mut problems = Vec::new();
        for (talib, p) in patterns {
            let name = talib.to_lowercase();
            let lb = p[lookback_key].as_u64().unwrap() as usize;
            if lookback_with_settings(&name, &settings) != Some(lb) {
                problems.push(format!(
                    "{talib}: lookback {:?}, TA-Lib {lb}",
                    lookback_with_settings(&name, &settings)
                ));
                continue;
            }
            let mut want = vec![0.0; r.bars.len()];
            for e in p[which].as_array().unwrap() {
                want[e[0].as_u64().unwrap() as usize] = e[1].as_f64().unwrap();
            }
            let got = run(&name, settings, &r.bars);
            if !nan_at(&got, 0, lb) {
                problems.push(format!("{talib}: not NaN before the lookback {lb}"));
            }
            let bad: Vec<_> = (lb..got.len()).filter(|&i| got[i] != want[i]).collect();
            if !bad.is_empty() {
                let first: Vec<_> = bad.iter().take(3).map(|&i| (i, got[i], want[i])).collect();
                problems.push(format!(
                    "{talib}: {} bars differ, first (bar, ours, TA-Lib) {first:?}",
                    bad.len()
                ));
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn default_settings_replay_ta_lib() {
        let r = replay();
        assert_eq!(table(&r.json["settings_default"]), CandleSettings::TALIB);
        replay_all(&r, CandleSettings::TALIB, "default", "lookback_default");
    }

    #[test]
    fn changed_settings_replay_ta_lib() {
        // BodyLong factor 1.5, BodyDoji period 5 and factor 0.2, Near factor 0.4.
        let r = replay();
        let changed = table(&r.json["settings_changed"]);
        assert_eq!(
            changed,
            CandleSettings::TALIB
                .with(BL, setting(RangeType::RealBody, 10, 1.5))
                .with(BD, setting(RangeType::HighLow, 5, 0.2))
                .with(NR, setting(RangeType::HighLow, 5, 0.4))
        );
        replay_all(&r, changed, "changed", "lookback_changed");
    }

    #[test]
    fn registry_warmup_is_the_first_valid_bar() {
        let r = replay();
        for d in all().filter(|d| d.family == Family::Candles) {
            let out = run(d.name, CandleSettings::TALIB, &r.bars);
            let first = out.iter().position(|v| !v.is_nan());
            assert_eq!(first, Some((d.warmup)(&defaults(d.name))), "{}", d.name);
        }
    }
}
