//! Trading-session calendars: when an instrument can trade.
//!
//! A [`SessionSchedule`] answers "is this instant inside a trading window?"
//! for one venue. It holds a UTC offset, trading windows for each weekday
//! (several per day, so a lunch break is two windows), a set of holiday dates
//! supplied by the caller, and an optional square-off minute for intraday
//! positions.
//!
//! No holiday list ships with the crate. Exchange calendars change every year
//! and are published by the exchange; the caller passes the dates it trusts.
//!
//! All calendar arithmetic is integer arithmetic on days since the Unix epoch
//! (the civil-date algorithms of H. Hinnant), so there is no date dependency
//! and no floating-point rounding at a day boundary.
//!
//! Conventions:
//!
//! * A window is half-open: `open <= t < close`. A print stamped exactly at
//!   the close is outside the session; one stamped exactly at the open is in.
//! * Minutes are local minutes from midnight (`9 * 60 + 15` is 09:15).
//! * Dates are `yyyymmdd` integers (`20260115`).
//! * Weekday windows are indexed Monday = 0 through Sunday = 6.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::core::error::RaptorError;
use crate::core::Result;

const NS_PER_MINUTE: i64 = 60_000_000_000;
const NS_PER_DAY: i64 = 86_400_000_000_000;
const MINUTES_PER_DAY: u32 = 24 * 60;

/// India Standard Time, UTC+5:30, in nanoseconds.
const IST_OFFSET_NS: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;

/// How many days `next_open_ns` looks ahead before giving up, so a schedule
/// whose every day is a holiday ends the search instead of looping.
const NEXT_OPEN_HORIZON_DAYS: i64 = 400;

/// The first date (`yyyymmdd`) the equity-derivatives segments close at 15:40.
pub const NFO_CLOSE_1540_FROM: i32 = 20260803;

/// The first date (`yyyymmdd`) the equity-derivatives segments close at 15:45 (the closing
/// auction session 2.0 of the SEBI circular of 2026-10-15).
pub const NFO_CLOSE_1545_FROM: i32 = 20261207;

/// Days since 1970-01-01 for a proleptic Gregorian date.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The `(year, month, day)` of a day count since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Day count for a `yyyymmdd` date, or `None` if it is not a real date in
/// 1900..=2200.
pub fn day_from_yyyymmdd(date: i32) -> Option<i64> {
    let (year, month, day) = (date / 10_000, (date / 100) % 100, date % 100);
    if !(1900..=2200).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let days = days_from_civil(year as i64, month as u32, day as u32);
    // Round-trip rejects 20260231 and friends.
    (civil_from_days(days) == (year as i64, month as u32, day as u32)).then_some(days)
}

/// `yyyymmdd` for a day count since 1970-01-01.
pub fn yyyymmdd_from_day(day: i64) -> i32 {
    let (y, m, d) = civil_from_days(day);
    (y * 10_000 + m as i64 * 100 + d as i64) as i32
}

/// Monday = 0 .. Sunday = 6 for a day count (1970-01-01 was a Thursday).
fn weekday_of(day: i64) -> usize {
    (day + 3).rem_euclid(7) as usize
}

/// Trading windows that replace the base ones from one local day to another
/// (a venue moving its close, as the equity-derivatives segment did on
/// 2026-08-03).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Era {
    /// First local day number the windows apply to.
    from: i64,
    /// Last local day number the windows apply to.
    to: i64,
    windows: [Vec<(u32, u32)>; 7],
}

/// The classes MCX trades to different hours. The fee classes derive from the same table
/// (agricultural or not), so a name has one class for its session and its charges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McxClass {
    /// Cotton, Kapas and Cotton Seed Wash Oil: Mon-Fri 09:00-21:00.
    SelectAgri,
    /// Every other agricultural contract: 09:00-17:00.
    OtherAgri,
    /// Bullion, base metals and energy: the evening session of [`SessionSchedule::mcx`].
    Standard,
}

/// An MCX contract whose name is in no class of the table, so no session is implied for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownMcxClass(pub String);

impl std::fmt::Display for UnknownMcxClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no session is implied for the MCX contract {:?}: its class is not known. Select \
             agricultural contracts (COTTON, KAPAS, COTWASOIL) trade 09:00-21:00, every other \
             agricultural contract 09:00-17:00, and bullion, base metals and energy contracts \
             follow SessionSchedule.mcx(); pass session= (SessionSchedule.mcx_select_agri(), \
             .mcx_other_agri() or .mcx()) for this contract",
            self.0
        )
    }
}

/// Which class an MCX contract name (an underlying or a symbol such as `COTTON26DECFUT`, or
/// either as an `EXCHANGE:NAME` key, `MCX:COTTON26DECFUT`) belongs to, or `None` for a name the
/// table does not list.
pub fn mcx_class(name: &str) -> Option<McxClass> {
    const SELECT: [&str; 3] = ["COTTON", "KAPAS", "COTWASOIL"];
    const OTHER: [&str; 19] = [
        "CPO",
        "RBDPMOLEIN",
        "MENTHA",
        "CASTOR",
        "CARDAMOM",
        "GUAR",
        "SOYBEAN",
        "SOYOIL",
        "REFSOY",
        "MUSTARD",
        "JEERA",
        "TURMERIC",
        "CORIANDER",
        "RUBBER",
        "SUGAR",
        "WHEAT",
        "MAIZE",
        "BARLEY",
        "COCUDAKL",
    ];
    const STANDARD: [&str; 19] = [
        "GOLD",
        "SILVER",
        "COPPER",
        "ALUMINIUM",
        "ALUMINI",
        "ZINC",
        "LEAD",
        "NICKEL",
        "CRUDEOIL",
        "NATURALGAS",
        "NATGAS",
        "BULLDEX",
        "METLDEX",
        "ENERGYDEX",
        "MCXBULLDEX",
        "MCXMETLDEX",
        "MCXENRGDEX",
        "MCXCOMPDEX",
        "GOLDPETAL",
    ];
    let upper = name.trim().to_ascii_uppercase();
    let upper = upper.rsplit(':').next().unwrap_or_default(); // a key names its exchange first
    let upper = upper.trim_start_matches(|c: char| !c.is_ascii_alphabetic());
    let starts = |table: &[&str]| table.iter().any(|p| upper.starts_with(p));
    if starts(&SELECT) {
        Some(McxClass::SelectAgri)
    } else if starts(&OTHER) {
        Some(McxClass::OtherAgri)
    } else if starts(&STANDARD) {
        Some(McxClass::Standard)
    } else {
        None
    }
}

/// The first Sunday of a month plus seven days for each further one: `n` = 1 is the first Sunday.
fn nth_sunday(year: i64, month: u32, n: i64) -> i64 {
    let first = days_from_civil(year, month, 1);
    first + (6 - weekday_of(first) as i64).rem_euclid(7) + 7 * (n - 1)
}

/// Whether the US is on daylight saving time on a local day, as the exchange follows it: from the
/// first trading day after the second Sunday of March to the last one before the first Sunday of
/// November's next trading day (the Monday after each change).
fn us_dst_in_force(day: i64) -> bool {
    let (year, _, _) = civil_from_days(day);
    nth_sunday(year, 3, 2) < day && day <= nth_sunday(year, 11, 1)
}

/// One venue's trading calendar.
///
/// What trades on a local date is decided in this order: a per-date
/// override (a special session, a half day, Muhurat trading; an empty
/// override closes the date), then the holiday set, then the latest dated
/// window change that covers the date, then the base weekday windows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSchedule {
    tz_offset_ns: i64,
    /// `(open_minute, close_minute)` windows per weekday, Monday first.
    windows: [Vec<(u32, u32)>; 7],
    holidays: BTreeSet<i32>,
    squareoff_minute: Option<u32>,
    /// Dated window changes, in the order they were added; the last one that
    /// covers a date answers for it.
    #[serde(default)]
    eras: Vec<Era>,
    /// Windows of single dates (`yyyymmdd`), which beat holidays and eras.
    #[serde(default)]
    overrides: BTreeMap<i32, Vec<(u32, u32)>>,
    /// Local day ranges (inclusive) the schedule answers for. Empty means
    /// every date; a schedule loaded from calendar documents answers only for
    /// the dates the documents cover and treats any other date as closed
    /// rather than guessing.
    #[serde(default)]
    covered: Vec<(i64, i64)>,
    /// A standing rule: while the US is on daylight saving time (`us_dst_in_force`) these weekday
    /// windows replace the base ones, after overrides, holidays and dated changes. For a venue
    /// whose hours follow the US clocks.
    #[serde(default)]
    us_summer: Option<[Vec<(u32, u32)>; 7]>,
}

impl SessionSchedule {
    /// Build and validate a schedule.
    ///
    /// Errors name the offending field: an offset outside +-24h, a window that
    /// does not open before it closes, overlapping windows on one day, a bad
    /// holiday date, a square-off minute outside every window, or a schedule
    /// with no trading window at all.
    pub fn new(
        tz_offset_ns: i64,
        windows: [Vec<(u32, u32)>; 7],
        holidays: impl IntoIterator<Item = i32>,
        squareoff_minute: Option<u32>,
    ) -> Result<Self> {
        if tz_offset_ns.abs() >= NS_PER_DAY {
            return Err(RaptorError::invalid_parameter(format!(
                "tz_offset_ns must be within +-24 hours, got {tz_offset_ns}"
            )));
        }
        let mut windows = windows;
        Self::check_windows(&mut windows)?;
        if windows.iter().all(Vec::is_empty) {
            return Err(RaptorError::invalid_parameter(
                "a session schedule needs at least one trading window",
            ));
        }
        let holidays = Self::checked_holidays(holidays)?;
        let schedule = Self {
            tz_offset_ns,
            windows,
            holidays,
            squareoff_minute: None,
            eras: Vec::new(),
            overrides: BTreeMap::new(),
            covered: Vec::new(),
            us_summer: None,
        };
        schedule.with_squareoff_minute(squareoff_minute)
    }

    /// Sort and validate one set of weekday windows.
    fn check_windows(windows: &mut [Vec<(u32, u32)>; 7]) -> Result<()> {
        const NAMES: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
        for (weekday, day_windows) in windows.iter_mut().enumerate() {
            Self::check_day(NAMES[weekday], day_windows)?;
        }
        Ok(())
    }

    fn check_day(name: &str, day_windows: &mut [(u32, u32)]) -> Result<()> {
        day_windows.sort_unstable();
        let mut previous_close = 0;
        for &(open, close) in day_windows.iter() {
            if close > MINUTES_PER_DAY {
                return Err(RaptorError::invalid_parameter(format!(
                    "{name} window ({open}, {close}): close minute must be <= {MINUTES_PER_DAY}"
                )));
            }
            if open >= close {
                return Err(RaptorError::invalid_parameter(format!(
                    "{name} window ({open}, {close}): open minute must be before close minute"
                )));
            }
            if open < previous_close {
                return Err(RaptorError::invalid_parameter(format!(
                    "{name} windows overlap at minute {open}"
                )));
            }
            previous_close = close;
        }
        Ok(())
    }

    fn checked_holidays(dates: impl IntoIterator<Item = i32>) -> Result<BTreeSet<i32>> {
        let mut set = BTreeSet::new();
        for date in dates {
            if day_from_yyyymmdd(date).is_none() {
                return Err(RaptorError::invalid_parameter(format!(
                    "holiday {date} is not a yyyymmdd date between 1900 and 2200"
                )));
            }
            set.insert(date);
        }
        Ok(set)
    }

    /// The same schedule with exactly these holidays (replacing any set
    /// before). Dates are `yyyymmdd`.
    pub fn with_holidays(mut self, dates: impl IntoIterator<Item = i32>) -> Result<Self> {
        self.holidays = Self::checked_holidays(dates)?;
        Ok(self)
    }

    /// The same schedule with another square-off minute (`None` disables it).
    ///
    /// The minute must lie inside a trading window or at its close: a
    /// square-off that can never be reached is a typo, not a setting.
    pub fn with_squareoff_minute(mut self, minute: Option<u32>) -> Result<Self> {
        if let Some(m) = minute {
            let reachable = self
                .windows
                .iter()
                .chain(self.eras.iter().flat_map(|e| e.windows.iter()))
                .chain(self.us_summer.iter().flatten())
                .flatten()
                .chain(self.overrides.values().flatten())
                .any(|&(open, close)| open <= m && m <= close);
            if !reachable {
                return Err(RaptorError::invalid_parameter(format!(
                    "squareoff_minute {m} is outside every trading window"
                )));
            }
        }
        self.squareoff_minute = minute;
        Ok(self)
    }

    /// Monday-to-Friday, one `[open, close)` window, at an offset.
    fn weekdays(
        tz_offset_ns: i64,
        open: u32,
        close: u32,
        squareoff_minute: Option<u32>,
    ) -> Result<Self> {
        Self::weekdays_of(tz_offset_ns, vec![(open, close)], squareoff_minute)
    }

    /// Monday-to-Friday with the same windows every day.
    fn weekdays_of(
        tz_offset_ns: i64,
        day: Vec<(u32, u32)>,
        squareoff_minute: Option<u32>,
    ) -> Result<Self> {
        Self::new(tz_offset_ns, Self::monday_to_friday(day), [], squareoff_minute)
    }

    fn monday_to_friday(day: Vec<(u32, u32)>) -> [Vec<(u32, u32)>; 7] {
        [day.clone(), day.clone(), day.clone(), day.clone(), day, vec![], vec![]]
    }

    /// NSE cash equities: Mon-Fri 09:15-15:30 IST.
    ///
    /// The regular session only; the pre-open and post-close auctions are not
    /// modelled. No square-off minute: a position is squared off only when the
    /// strategy declares intraday holding or the run asks for it (see
    /// [`SessionSchedule::with_squareoff_minute`]). No holiday list ships.
    pub fn nse_equity() -> Self {
        Self::weekdays(IST_OFFSET_NS, 9 * 60 + 15, 15 * 60 + 30, None).expect("preset is valid")
    }

    /// NSE equity derivatives: Mon-Fri 09:15-15:45 IST from 2026-12-07, 09:15-15:40 from
    /// 2026-08-03 (the NSE market-timings page), 09:15-15:30 before (the Closing Auction
    /// Session for F&O stocks moves the close; the cash segment keeps 15:30).
    pub fn nse_fo() -> Self {
        Self::nse_equity()
            .with_window_change(
                NFO_CLOSE_1540_FROM,
                None,
                Self::monday_to_friday(vec![(9 * 60 + 15, 15 * 60 + 40)]),
            )
            .and_then(|s| {
                s.with_window_change(
                    NFO_CLOSE_1545_FROM,
                    None,
                    Self::monday_to_friday(vec![(9 * 60 + 15, 15 * 60 + 45)]),
                )
            })
            .expect("preset is valid")
    }

    /// BSE cash equities: Mon-Fri 09:15-15:30 IST.
    /// The preset of an Indian exchange segment, by the venue code an `InstrumentSpec` names, or
    /// `None` for a venue no preset covers: such an instrument has no session until one is given.
    pub fn for_venue(venue: &str) -> Option<Self> {
        match venue.to_ascii_uppercase().as_str() {
            "NSE" => Some(Self::nse_equity()),
            "NFO" => Some(Self::nse_fo()),
            "BSE" => Some(Self::bse_equity()),
            "BFO" => Some(Self::bse_fo()),
            "MCX" => Some(Self::mcx()),
            "CDS" => Some(Self::cds()),
            "BCD" => Some(Self::bcd()),
            _ => None,
        }
    }

    /// The preset of a venue for one instrument, by the names it carries (its underlying, then its
    /// symbol). On MCX the contract's class picks the session: Cotton, Kapas and Cotton Seed Wash
    /// Oil close at 21:00, every other agricultural contract at 17:00, bullion, base metals and
    /// energy follow [`SessionSchedule::mcx`]. A name in none of those classes is not guessed: it
    /// is an [`UnknownMcxClass`] error, so the caller gives `session=`. Any other venue is
    /// [`SessionSchedule::for_venue`].
    pub fn for_instrument(
        venue: &str,
        names: &[&str],
    ) -> std::result::Result<Option<Self>, UnknownMcxClass> {
        if !venue.eq_ignore_ascii_case("MCX") {
            return Ok(Self::for_venue(venue));
        }
        match names.iter().find_map(|n| mcx_class(n)) {
            Some(McxClass::SelectAgri) => Ok(Some(Self::mcx_select_agri())),
            Some(McxClass::OtherAgri) => Ok(Some(Self::mcx_other_agri())),
            Some(McxClass::Standard) => Ok(Some(Self::mcx())),
            None => Err(UnknownMcxClass(names.first().copied().unwrap_or_default().to_string())),
        }
    }

    pub fn bse_equity() -> Self {
        Self::nse_equity()
    }

    /// BSE equity derivatives: as [`SessionSchedule::nse_fo`].
    pub fn bse_fo() -> Self {
        Self::nse_fo()
    }

    /// MCX non-agricultural contracts: Mon-Fri, a day session 09:00-17:00 and
    /// an evening session from 17:00 that closes at 23:55 IST while the US is
    /// on standard time and at 23:30 while it is on daylight saving.
    ///
    /// The close follows a standing rule with no end: it moves on the first trading day after
    /// the US clocks change, the Monday after the second Sunday of March (23:30) and the Monday
    /// after the first Sunday of November (23:55). That reproduces the changes the exchange
    /// named: 2024-11-04, 2025-03-10, 2025-11-03, 2026-03-09 and 2026-11-02. Coverage starts at
    /// 2024-11-04, the first of them; a bar before it raises `session_not_covered` (the market is
    /// not known closed there), so load a calendar document. This is the normal session, not a
    /// holiday calendar.
    pub fn mcx() -> Self {
        Self::mcx_closing(vec![(9 * 60, 17 * 60), (17 * 60, 23 * 60 + 55)], Some(23 * 60 + 30))
    }

    /// MCX select agricultural contracts (Cotton, Cotton Seed Wash Oil, Kapas): the same days, closing at 21:00 all year.
    pub fn mcx_select_agri() -> Self {
        Self::mcx_closing(vec![(9 * 60, 17 * 60), (17 * 60, 21 * 60)], None)
    }

    /// All other MCX agricultural contracts: 09:00-17:00 all year.
    pub fn mcx_other_agri() -> Self {
        Self::mcx_closing(vec![(9 * 60, 17 * 60)], None)
    }

    /// Covered from the exchange's first named change, with no end.
    fn mcx_closing(winter: Vec<(u32, u32)>, summer_close: Option<u32>) -> Self {
        let mut schedule =
            Self::new(IST_OFFSET_NS, Self::monday_to_friday(winter.clone()), [], None)
                .and_then(|s| s.with_coverage(20241104, 22001231))
                .expect("preset is valid");
        if let Some(close) = summer_close {
            let mut summer = winter;
            if let Some(last) = summer.last_mut() {
                last.1 = close;
            }
            schedule.us_summer = Some(Self::monday_to_friday(summer));
        }
        schedule
    }

    /// Trading minutes in a regular session: the longest weekday of the base
    /// windows (what annualisation counts per day).
    pub fn trading_minutes(&self) -> f64 {
        self.windows
            .iter()
            .map(|day| day.iter().map(|&(o, c)| f64::from(c - o)).sum::<f64>())
            .fold(0.0, f64::max)
    }

    /// The same schedule answering only for local dates `from` to `to`
    /// (`yyyymmdd`, inclusive); every other date is closed.
    pub fn with_coverage(mut self, from: i32, to: i32) -> Result<Self> {
        let bad = |d: i32| {
            RaptorError::invalid_parameter(format!(
                "{d} is not a yyyymmdd date between 1900 and 2200"
            ))
        };
        let (a, b) = (
            day_from_yyyymmdd(from).ok_or_else(|| bad(from))?,
            day_from_yyyymmdd(to).ok_or_else(|| bad(to))?,
        );
        if b < a {
            return Err(RaptorError::invalid_parameter(format!(
                "coverage {from}..{to} ends before it starts"
            )));
        }
        self.covered = vec![(a, b)];
        Ok(self)
    }

    /// The local date ranges (`yyyymmdd`, inclusive) the schedule answers
    /// for; empty means every date.
    pub fn coverage(&self) -> Vec<(i32, i32)> {
        self.covered.iter().map(|&(a, b)| (yyyymmdd_from_day(a), yyyymmdd_from_day(b))).collect()
    }

    /// NSE currency derivatives on INR pairs: Mon-Fri 09:00-17:00 IST.
    pub fn cds() -> Self {
        Self::weekdays(IST_OFFSET_NS, 9 * 60, 17 * 60, None).expect("preset is valid")
    }

    /// NSE currency derivatives on cross-currency pairs: Mon-Fri 09:00-19:30 IST.
    pub fn cds_cross() -> Self {
        Self::weekdays(IST_OFFSET_NS, 9 * 60, 19 * 60 + 30, None).expect("preset is valid")
    }

    /// Currency derivatives on the BSE: as [`SessionSchedule::cds`].
    pub fn bcd() -> Self {
        Self::cds()
    }

    /// A schedule for one venue from dated calendar documents.
    ///
    /// Each document is JSON:
    ///
    /// ```json
    /// {
    ///   "name": "calendar-2026-q4",
    ///   "utc_offset": "+05:30",
    ///   "valid_from": "2026-09-28",
    ///   "valid_to": "2026-12-31",
    ///   "sessions": {
    ///     "NSE": {"open": "09:15", "close": "15:30"},
    ///     "MCX": {"windows": [["09:00", "17:00"], ["17:00", "23:55"]]}
    ///   },
    ///   "holidays": [{"date": "2026-10-02", "venues": ["NSE", "MCX"]}],
    ///   "overrides": [
    ///     {"venue": "MCX", "date": "2026-10-20", "open": "17:00", "close": "23:55"},
    ///     {"venue": "NSE", "date": "2026-11-08", "windows": [["18:00", "19:00"]]}
    ///   ]
    /// }
    /// ```
    ///
    /// `sessions` windows apply Monday to Friday (add `"weekdays": ["mon",
    /// ..]` to change that). `holidays` and `overrides` are optional and
    /// supplied by the caller: the crate ships no holiday list. An override
    /// opens a date whatever the weekday or holiday set says; `"closed":
    /// true` closes it. The venue named in `sessions` must exist in every
    /// document. Later documents answer for the dates they cover; a date no
    /// document covers is closed, not guessed. All documents must share one
    /// `utc_offset`. `venues` and `venue` also read as `exchanges` and
    /// `exchange`.
    pub fn from_calendars(venue: &str, documents: &[&str]) -> Result<Self> {
        calendar::from_calendars(venue, documents)
    }

    pub fn tz_offset_ns(&self) -> i64 {
        self.tz_offset_ns
    }

    /// Windows for a weekday, Monday = 0.
    pub fn windows(&self, weekday: usize) -> &[(u32, u32)] {
        &self.windows[weekday]
    }

    pub fn holidays(&self) -> impl Iterator<Item = i32> + '_ {
        self.holidays.iter().copied()
    }

    pub fn squareoff_minute(&self) -> Option<u32> {
        self.squareoff_minute
    }

    /// `(local day number, nanoseconds into that local day)`.
    #[inline]
    fn local(&self, ts_ns: i64) -> (i64, i64) {
        let local = ts_ns + self.tz_offset_ns;
        (local.div_euclid(NS_PER_DAY), local.rem_euclid(NS_PER_DAY))
    }

    /// Windows of a local day, or `None` when it is a holiday or has none.
    fn trading_windows(&self, day: i64) -> Option<&[(u32, u32)]> {
        if !self.covered.is_empty() && !self.covered.iter().any(|&(a, b)| a <= day && day <= b) {
            return None;
        }
        let date = yyyymmdd_from_day(day);
        if let Some(special) = self.overrides.get(&date) {
            return (!special.is_empty()).then_some(special.as_slice());
        }
        if self.holidays.contains(&date) {
            return None;
        }
        let weekday = weekday_of(day);
        let windows = self
            .eras
            .iter()
            .rev()
            .find(|e| e.from <= day && day <= e.to)
            .map(|e| &e.windows[weekday])
            .or_else(|| {
                self.us_summer.as_ref().filter(|_| us_dst_in_force(day)).map(|w| &w[weekday])
            })
            .unwrap_or(&self.windows[weekday]);
        (!windows.is_empty()).then_some(windows.as_slice())
    }

    /// The windows `(open_minute, close_minute)` of a local `yyyymmdd` date
    /// after overrides, holidays and dated changes; empty when it does not
    /// trade.
    pub fn windows_on(&self, date: i32) -> Vec<(u32, u32)> {
        day_from_yyyymmdd(date)
            .and_then(|day| self.trading_windows(day))
            .map(<[_]>::to_vec)
            .unwrap_or_default()
    }

    /// The trading windows of the local day of `ts_ns` as UTC nanoseconds
    /// `(open, close)`, in order; empty when the day does not trade.
    pub fn windows_ns_on_day_of(&self, ts_ns: i64) -> Vec<(i64, i64)> {
        let (day, _) = self.local(ts_ns);
        self.trading_windows(day)
            .map(|ws| {
                ws.iter()
                    .map(|&(o, c)| {
                        let base = day * NS_PER_DAY - self.tz_offset_ns;
                        (base + o as i64 * NS_PER_MINUTE, base + c as i64 * NS_PER_MINUTE)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The same schedule with other weekday windows from one local date to
    /// another (both `yyyymmdd`, inclusive; `to` of `None` is open-ended). A
    /// later change beats an earlier one where they overlap.
    pub fn with_window_change(
        mut self,
        from: i32,
        to: Option<i32>,
        windows: [Vec<(u32, u32)>; 7],
    ) -> Result<Self> {
        let bad = |d: i32| {
            RaptorError::invalid_parameter(format!(
                "{d} is not a yyyymmdd date between 1900 and 2200"
            ))
        };
        let from_day = day_from_yyyymmdd(from).ok_or_else(|| bad(from))?;
        let to_day = match to {
            Some(t) => day_from_yyyymmdd(t).ok_or_else(|| bad(t))?,
            None => i64::MAX,
        };
        if to_day < from_day {
            return Err(RaptorError::invalid_parameter(format!(
                "window change from {from} ends before it starts"
            )));
        }
        let mut windows = windows;
        Self::check_windows(&mut windows)?;
        self.eras.push(Era { from: from_day, to: to_day, windows });
        Ok(self)
    }

    /// The same schedule with these windows on one local date (`yyyymmdd`),
    /// whatever the weekday or holiday set says: a special session, a half
    /// day, Muhurat trading. Empty windows close the date.
    pub fn with_session_override(
        mut self,
        date: i32,
        mut windows: Vec<(u32, u32)>,
    ) -> Result<Self> {
        if day_from_yyyymmdd(date).is_none() {
            return Err(RaptorError::invalid_parameter(format!(
                "override date {date} is not a yyyymmdd date between 1900 and 2200"
            )));
        }
        Self::check_day(&date.to_string(), &mut windows)?;
        self.overrides.insert(date, windows);
        Ok(self)
    }

    /// Per-date overrides, as `(yyyymmdd, windows)` in date order.
    pub fn session_overrides(&self) -> impl Iterator<Item = (i32, &[(u32, u32)])> + '_ {
        self.overrides.iter().map(|(d, w)| (*d, w.as_slice()))
    }

    /// Whether `ts_ns` falls inside a trading window (`open <= t < close`).
    pub fn is_open(&self, ts_ns: i64) -> bool {
        let (day, tod) = self.local(ts_ns);
        self.trading_windows(day).is_some_and(|windows| {
            windows.iter().any(|&(open, close)| {
                open as i64 * NS_PER_MINUTE <= tod && tod < close as i64 * NS_PER_MINUTE
            })
        })
    }

    /// Local trading day number of `ts_ns`, or `None` on a weekend, a
    /// holiday or any day without a window. Time of day is ignored, so a
    /// print before the open or after the close still belongs to its day.
    pub(crate) fn session_day(&self, ts_ns: i64) -> Option<i64> {
        let (day, _) = self.local(ts_ns);
        self.trading_windows(day).map(|_| day)
    }

    /// `yyyymmdd` of the trading day `ts_ns` falls on; see
    /// [`SessionSchedule::session_day`] for what has no session.
    pub fn session_date(&self, ts_ns: i64) -> Option<i32> {
        self.session_day(ts_ns).map(yyyymmdd_from_day)
    }

    /// UTC nanoseconds of the last close on a trading date, or `None` when
    /// the date is not a real date or is not a trading day.
    pub fn session_close_ns(&self, date: i32) -> Option<i64> {
        let day = day_from_yyyymmdd(date)?;
        self.close_of_day(day)
    }

    fn close_of_day(&self, day: i64) -> Option<i64> {
        let &(_, close) = self.trading_windows(day)?.last()?;
        Some(day * NS_PER_DAY + close as i64 * NS_PER_MINUTE - self.tz_offset_ns)
    }

    /// The first window-open instant strictly after `ts_ns`, or `None` if
    /// there is none within a year (every day a holiday).
    ///
    /// Inside a window this is the next window's open: after a lunch break,
    /// or the next trading day.
    pub fn next_open_ns(&self, ts_ns: i64) -> Option<i64> {
        let (first_day, _) = self.local(ts_ns);
        for day in first_day..first_day + NEXT_OPEN_HORIZON_DAYS {
            let Some(windows) = self.trading_windows(day) else { continue };
            for &(open, _) in windows {
                let open_ns = day * NS_PER_DAY + open as i64 * NS_PER_MINUTE - self.tz_offset_ns;
                if open_ns > ts_ns {
                    return Some(open_ns);
                }
            }
        }
        None
    }

    /// When a DAY order placed at `submitted_ts_ns` lapses: the close of its
    /// session, or of the next session when it was placed after the close or
    /// on a non-trading day.
    pub(crate) fn day_order_expiry_ns(&self, submitted_ts_ns: i64) -> Option<i64> {
        let (first_day, _) = self.local(submitted_ts_ns);
        (first_day..first_day + NEXT_OPEN_HORIZON_DAYS)
            .filter_map(|day| self.close_of_day(day))
            .find(|&close| close > submitted_ts_ns)
    }

    /// The trading day whose square-off is due at `ts_ns`, or `None`.
    ///
    /// Due on the first event at or after the square-off minute on a trading
    /// day. `last_done` is the day number last returned; passing it back
    /// makes the answer fire once per day, which is the contract of
    /// `squareoff_flags` applied to a calendar with holidays and weekends.
    pub(crate) fn squareoff_due(&self, ts_ns: i64, last_done: Option<i64>) -> Option<i64> {
        self.squareoff_due_at(self.squareoff_minute?, ts_ns, last_done)
    }

    /// [`SessionSchedule::squareoff_due`] at a local minute the caller names
    /// (a run's `squareoff_time`), read in this schedule's time zone.
    pub(crate) fn squareoff_due_at(
        &self,
        minute: u32,
        ts_ns: i64,
        last_done: Option<i64>,
    ) -> Option<i64> {
        let (day, tod) = self.local(ts_ns);
        if tod < minute as i64 * NS_PER_MINUTE || last_done == Some(day) {
            return None;
        }
        self.trading_windows(day).map(|_| day)
    }
}

mod calendar;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session::squareoff_flags;

    const HOUR: i64 = 3_600 * 1_000_000_000;
    const MIN: i64 = NS_PER_MINUTE;

    /// UTC ns of an IST wall-clock time.
    fn ist(date: i32, hour: i64, minute: i64) -> i64 {
        day_from_yyyymmdd(date).unwrap() * NS_PER_DAY + hour * HOUR + minute * MIN - IST_OFFSET_NS
    }

    #[test]
    fn civil_dates_round_trip_and_know_the_weekday() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(weekday_of(0), 3); // Thursday
        assert_eq!(day_from_yyyymmdd(20260115), Some(days_from_civil(2026, 1, 15)));
        assert_eq!(yyyymmdd_from_day(day_from_yyyymmdd(20240229).unwrap()), 20240229);
        // 2026-01-15 is a Thursday, 2026-01-17 a Saturday.
        assert_eq!(weekday_of(day_from_yyyymmdd(20260115).unwrap()), 3);
        assert_eq!(weekday_of(day_from_yyyymmdd(20260117).unwrap()), 5);
        assert_eq!(day_from_yyyymmdd(20260231), None);
        assert_eq!(day_from_yyyymmdd(20251301), None);
        assert_eq!(day_from_yyyymmdd(20230229), None);
        assert!(day_from_yyyymmdd(20240229).is_some());
    }

    #[test]
    fn nse_window_is_half_open_and_weekday_only() {
        let s = SessionSchedule::nse_equity();
        assert!(!s.is_open(ist(20260115, 9, 14)));
        assert!(s.is_open(ist(20260115, 9, 15)));
        assert!(s.is_open(ist(20260115, 15, 29)));
        assert!(!s.is_open(ist(20260115, 15, 30)));
        assert!(!s.is_open(ist(20260117, 10, 0)), "Saturday");
        assert!(!s.is_open(ist(20260118, 10, 0)), "Sunday");
        assert!(s.is_open(ist(20260119, 10, 0)), "Monday");
    }

    #[test]
    fn timezone_decides_the_local_day() {
        // 03:45 UTC is 09:15 IST; the same instant is closed in UTC.
        let utc_instant = days_from_civil(2026, 1, 15) * NS_PER_DAY + 3 * HOUR + 45 * MIN;
        assert!(SessionSchedule::nse_equity().is_open(utc_instant));
        let utc = SessionSchedule::weekdays(0, 9 * 60 + 15, 15 * 60 + 30, None).unwrap();
        assert!(!utc.is_open(utc_instant));
        // Negative offset (New York winter): 14:30 UTC is 09:30 local.
        let ny = SessionSchedule::weekdays(-5 * HOUR, 9 * 60 + 30, 16 * 60, None).unwrap();
        assert!(ny.is_open(days_from_civil(2026, 1, 15) * NS_PER_DAY + 14 * HOUR + 30 * MIN));
        assert!(!ny.is_open(days_from_civil(2026, 1, 15) * NS_PER_DAY + 14 * HOUR + 29 * MIN));
    }

    #[test]
    fn a_lunch_break_is_a_gap_between_two_windows() {
        let day = vec![(9 * 60, 12 * 60), (13 * 60, 15 * 60)];
        let s = SessionSchedule::new(
            0,
            [day.clone(), day.clone(), day.clone(), day.clone(), day, vec![], vec![]],
            [],
            None,
        )
        .unwrap();
        let at = |h, m| ist(20260115, h, m) + IST_OFFSET_NS;
        assert!(s.is_open(at(11, 59)));
        assert!(!s.is_open(at(12, 0)));
        assert!(!s.is_open(at(12, 59)));
        assert!(s.is_open(at(13, 0)));
        assert_eq!(s.next_open_ns(at(10, 0)), Some(at(13, 0)));
        assert_eq!(s.next_open_ns(at(12, 30)), Some(at(13, 0)));
        assert_eq!(s.session_close_ns(20260115), Some(at(15, 0)));
    }

    #[test]
    fn holidays_close_the_day_and_are_caller_supplied() {
        let plain = SessionSchedule::nse_equity();
        assert!(plain.is_open(ist(20260126, 10, 0)), "no calendar ships with the crate");
        let s = plain.with_holidays([20260126]).unwrap();
        assert!(!s.is_open(ist(20260126, 10, 0)));
        assert_eq!(s.session_date(ist(20260126, 10, 0)), None);
        assert_eq!(s.session_close_ns(20260126), None);
        // Monday 26th is a holiday, so the open after Friday's close is Tuesday's.
        assert_eq!(s.next_open_ns(ist(20260123, 15, 30)), Some(ist(20260127, 9, 15)));
        // with_holidays replaces rather than accumulates.
        let replaced = s.with_holidays([20260127]).unwrap();
        assert!(replaced.is_open(ist(20260126, 10, 0)));
        assert!(!replaced.is_open(ist(20260127, 10, 0)));
    }

    #[test]
    fn session_date_ignores_time_of_day_but_not_weekends() {
        let s = SessionSchedule::nse_equity();
        assert_eq!(s.session_date(ist(20260115, 3, 0)), Some(20260115));
        assert_eq!(s.session_date(ist(20260115, 23, 59)), Some(20260115));
        assert_eq!(s.session_date(ist(20260117, 12, 0)), None);
    }

    #[test]
    fn next_open_is_strictly_after() {
        let s = SessionSchedule::nse_equity();
        assert_eq!(s.next_open_ns(ist(20260115, 8, 0)), Some(ist(20260115, 9, 15)));
        assert_eq!(s.next_open_ns(ist(20260115, 9, 15)), Some(ist(20260116, 9, 15)));
        assert_eq!(s.next_open_ns(ist(20260116, 16, 0)), Some(ist(20260119, 9, 15)));
    }

    #[test]
    fn next_open_gives_up_when_every_day_is_a_holiday() {
        let first = day_from_yyyymmdd(20260101).unwrap();
        let all = (first - 5..first + 500).map(yyyymmdd_from_day);
        let s = SessionSchedule::nse_equity().with_holidays(all).unwrap();
        assert_eq!(s.next_open_ns(ist(20260105, 10, 0)), None);
    }

    #[test]
    fn presets_match_their_documented_times() {
        for s in [
            SessionSchedule::nse_equity(),
            SessionSchedule::nse_fo(),
            SessionSchedule::bse_equity(),
        ] {
            assert_eq!(s.tz_offset_ns(), IST_OFFSET_NS);
            for weekday in 0..5 {
                assert_eq!(s.windows(weekday), &[(555, 930)]);
            }
            assert!(s.windows(5).is_empty() && s.windows(6).is_empty());
            assert_eq!(s.squareoff_minute(), None, "square-off is the run's choice");
            assert_eq!(s.holidays().count(), 0);
        }
    }

    #[test]
    fn validation_names_the_problem() {
        let msg = |r: Result<SessionSchedule>| r.unwrap_err().to_string();
        let ok = vec![(540, 900)];
        let week = |w: Vec<(u32, u32)>| [w, vec![], vec![], vec![], vec![], vec![], vec![]];
        assert!(msg(SessionSchedule::new(25 * HOUR, week(ok.clone()), [], None))
            .contains("tz_offset_ns"));
        assert!(
            msg(SessionSchedule::new(0, week(vec![(900, 540)]), [], None)).contains("before close")
        );
        assert!(msg(SessionSchedule::new(0, week(vec![(0, 1500)]), [], None)).contains("<= 1440"));
        assert!(msg(SessionSchedule::new(0, week(vec![(540, 700), (600, 900)]), [], None))
            .contains("overlap"));
        assert!(msg(SessionSchedule::new(0, week(vec![]), [], None)).contains("at least one"));
        assert!(
            msg(SessionSchedule::new(0, week(ok.clone()), [20260230], None)).contains("holiday")
        );
        assert!(msg(SessionSchedule::new(0, week(ok), [], Some(1000))).contains("squareoff_minute"));
    }

    #[test]
    fn day_orders_lapse_at_the_close_of_their_session() {
        let s = SessionSchedule::nse_equity();
        assert_eq!(s.day_order_expiry_ns(ist(20260115, 10, 0)), Some(ist(20260115, 15, 30)));
        // Placed after the close, or at the weekend: the next session's close.
        assert_eq!(s.day_order_expiry_ns(ist(20260115, 16, 0)), Some(ist(20260116, 15, 30)));
        assert_eq!(s.day_order_expiry_ns(ist(20260117, 10, 0)), Some(ist(20260119, 15, 30)));
    }

    #[test]
    fn squareoff_fires_once_per_trading_day_and_agrees_with_squareoff_flags() {
        let s = SessionSchedule::nse_equity().with_squareoff_minute(Some(15 * 60 + 20)).unwrap();
        let sq = s.squareoff_minute().unwrap();
        assert_eq!(sq, 15 * 60 + 20);
        let stamps: Vec<i64> = [
            ist(20260115, 15, 19),
            ist(20260115, 15, 20),
            ist(20260115, 15, 25),
            ist(20260116, 15, 19),
            ist(20260116, 15, 21),
        ]
        .to_vec();
        let flags = squareoff_flags(&stamps, sq, IST_OFFSET_NS);
        let mut last = None;
        let ours: Vec<bool> = stamps
            .iter()
            .map(|&t| match s.squareoff_due(t, last) {
                Some(day) => {
                    last = Some(day);
                    true
                }
                None => false,
            })
            .collect();
        assert_eq!(ours, flags);
        assert_eq!(ours, [false, true, false, false, true]);
    }

    #[test]
    fn squareoff_skips_holidays_and_weekends() {
        let s = SessionSchedule::nse_equity()
            .with_squareoff_minute(Some(15 * 60 + 20))
            .unwrap()
            .with_holidays([20260126])
            .unwrap();
        assert_eq!(s.squareoff_due(ist(20260126, 15, 21), None), None);
        assert_eq!(s.squareoff_due(ist(20260117, 15, 21), None), None);
        assert!(s.squareoff_due(ist(20260115, 15, 21), None).is_some());
        let no_squareoff = SessionSchedule::nse_equity().with_squareoff_minute(None).unwrap();
        assert_eq!(no_squareoff.squareoff_due(ist(20260115, 15, 21), None), None);
    }

    #[test]
    fn the_equity_derivatives_close_moves_to_1540_on_2026_08_03() {
        let s = SessionSchedule::nse_fo();
        // Friday 2026-07-31: 15:30 close. Monday 2026-08-03: 15:40.
        assert!(s.is_open(ist(20260731, 15, 29)));
        assert!(!s.is_open(ist(20260731, 15, 30)));
        assert!(s.is_open(ist(20260803, 15, 39)));
        assert!(!s.is_open(ist(20260803, 15, 40)));
        assert_eq!(s.session_close_ns(20260731), Some(ist(20260731, 15, 30)));
        assert_eq!(s.session_close_ns(20260803), Some(ist(20260803, 15, 40)));
        assert_eq!(s.windows_on(20260803), vec![(555, 940)]);
        // The cash segment kept 15:30.
        assert!(!SessionSchedule::nse_equity().is_open(ist(20260803, 15, 31)));
    }

    #[test]
    fn the_equity_derivatives_close_moves_to_1545_on_2026_12_07() {
        for s in [SessionSchedule::nse_fo(), SessionSchedule::bse_fo()] {
            // Friday 2026-12-04: still 15:40. Monday 2026-12-07: 15:45.
            assert!(s.is_open(ist(20261204, 15, 39)) && !s.is_open(ist(20261204, 15, 40)));
            assert!(s.is_open(ist(20261207, 15, 44)) && !s.is_open(ist(20261207, 15, 45)));
            assert_eq!(s.session_close_ns(20261204), Some(ist(20261204, 15, 40)));
            assert_eq!(s.session_close_ns(20261207), Some(ist(20261207, 15, 45)));
            assert_eq!(s.windows_on(20261207), vec![(555, 945)]);
        }
        // The cash segment keeps 15:30.
        assert!(!SessionSchedule::nse_equity().is_open(ist(20261207, 15, 31)));
    }

    #[test]
    fn mcx_follows_the_us_clock_rule_and_reproduces_every_change_the_exchange_named() {
        let s = SessionSchedule::mcx();
        // Winter 2024-25 (from 2024-11-04), summer 2025 (from 2025-03-10), winter, summer 2026, winter.
        assert!(s.is_open(ist(20241105, 23, 54)) && !s.is_open(ist(20241105, 23, 55)));
        assert!(s.is_open(ist(20250611, 23, 29)) && !s.is_open(ist(20250611, 23, 30)));
        assert!(s.is_open(ist(20251201, 23, 54)) && !s.is_open(ist(20251201, 23, 55)));
        assert!(s.is_open(ist(20260615, 23, 29)) && !s.is_open(ist(20260615, 23, 30)));
        assert!(s.is_open(ist(20261102, 23, 54)) && !s.is_open(ist(20261102, 23, 55)));
        // The first trading day of each named change, and the day before it.
        for (date, before, close) in [
            (20241104, 20241101, 23 * 60 + 55),
            (20250310, 20250307, 23 * 60 + 30),
            (20251103, 20251031, 23 * 60 + 55),
            (20260309, 20260306, 23 * 60 + 30),
            (20261102, 20261030, 23 * 60 + 55),
        ] {
            assert_eq!(s.windows_on(date).last().map(|w| w.1), Some(close), "{date}");
            if date != 20241104 {
                assert_ne!(s.windows_on(before).last().map(|w| w.1), Some(close), "{before}");
            }
        }
        assert_eq!(s.session_close_ns(20251201), Some(ist(20251201, 23, 55)));
        assert!(!s.is_open(ist(20251206, 12, 0)), "Saturday");
        // Before the first named change nothing is guessed; after the last one the rule goes on.
        assert!(!s.is_open(ist(20241101, 12, 0)));
        assert_eq!(s.coverage(), vec![(20241104, 22001231)]);
        // 2027: 14 March is the second Sunday of March, 7 November the first of November.
        assert_eq!(s.windows_on(20270312).last().map(|w| w.1), Some(23 * 60 + 55));
        assert_eq!(s.windows_on(20270315).last().map(|w| w.1), Some(23 * 60 + 30));
        assert_eq!(s.windows_on(20271105).last().map(|w| w.1), Some(23 * 60 + 30));
        assert_eq!(s.windows_on(20271108).last().map(|w| w.1), Some(23 * 60 + 55));
        // 2030: second Sunday of March is the 10th.
        assert_eq!(s.windows_on(20300308).last().map(|w| w.1), Some(23 * 60 + 55));
        assert_eq!(s.windows_on(20300311).last().map(|w| w.1), Some(23 * 60 + 30));
        // Internationally referenceable agricultural contracts close at 21:00 all year.
        let agri = SessionSchedule::mcx_select_agri();
        assert!(agri.is_open(ist(20251201, 20, 59)) && !agri.is_open(ist(20251201, 21, 0)));
        assert!(agri.is_open(ist(20260615, 20, 59)) && !agri.is_open(ist(20260615, 21, 0)));
        assert!(agri.is_open(ist(20290615, 20, 59)) && !agri.is_open(ist(20290615, 21, 0)));
        // The other agricultural contracts close at 17:00 all year.
        let other = SessionSchedule::mcx_other_agri();
        assert!(other.is_open(ist(20251201, 16, 59)) && !other.is_open(ist(20251201, 17, 0)));
        assert!(other.is_open(ist(20260615, 16, 59)) && !other.is_open(ist(20260615, 17, 0)));
    }

    #[test]
    fn an_mcx_contract_takes_the_session_of_its_class() {
        let at = |names: &[&str]| SessionSchedule::for_instrument("MCX", names).unwrap().unwrap();
        let close = |s: &SessionSchedule| s.windows_on(20251201).last().map(|w| w.1);
        assert_eq!(close(&at(&["CRUDEOIL", "CRUDEOIL26DECFUT"])), Some(23 * 60 + 55));
        assert_eq!(close(&at(&["COTTON26DECFUT"])), Some(21 * 60));
        assert_eq!(close(&at(&["COTWASOIL26DECFUT"])), Some(21 * 60));
        assert_eq!(close(&at(&["CPO"])), Some(17 * 60));
        assert_eq!(close(&at(&["RBDPMOLEIN"])), Some(17 * 60));
        assert_eq!(close(&at(&["KAPAS"])), Some(21 * 60));
        assert_eq!(close(&at(&["MENTHAOIL26DECFUT"])), Some(17 * 60));
        // A platform key names the exchange first: the class is the contract's.
        assert_eq!(close(&at(&["MCX:MENTHAOIL", "MCX:MENTHAOIL26DECFUT"])), Some(17 * 60));
        assert_eq!(close(&at(&["MCX:CRUDEOIL", "MCX:CRUDEOIL26DECFUT"])), Some(23 * 60 + 55));
        // A name in no class is not guessed, whatever its first letters look like.
        let err = SessionSchedule::for_instrument("MCX", &["X", "F"]).unwrap_err();
        assert!(err.to_string().contains("COTWASOIL") && err.to_string().contains("09:00-17:00"));
        // Another venue is not read by name.
        assert_eq!(
            SessionSchedule::for_instrument("NSE", &["COTTON"]),
            Ok(SessionSchedule::for_venue("NSE"))
        );
    }

    #[test]
    fn currency_sessions_differ_between_inr_and_cross_pairs() {
        assert!(SessionSchedule::cds().is_open(ist(20260615, 16, 59)));
        assert!(!SessionSchedule::cds().is_open(ist(20260615, 17, 0)));
        assert!(SessionSchedule::cds_cross().is_open(ist(20260615, 19, 29)));
        assert!(!SessionSchedule::cds_cross().is_open(ist(20260615, 19, 30)));
        assert_eq!(SessionSchedule::nse_equity().trading_minutes(), 375.0);
        assert_eq!(
            SessionSchedule::nse_fo().trading_minutes(),
            375.0,
            "the base windows; 385 from 2026-08-03"
        );
    }

    #[test]
    fn a_date_override_beats_the_weekday_and_the_holiday_set() {
        // Muhurat trading on a Sunday that is also listed as a holiday.
        let s = SessionSchedule::nse_equity()
            .with_holidays([20261108])
            .unwrap()
            .with_session_override(20261108, vec![(18 * 60, 19 * 60)])
            .unwrap()
            .with_session_override(20260126, vec![])
            .unwrap();
        assert!(!s.is_open(ist(20261108, 12, 0)));
        assert!(s.is_open(ist(20261108, 18, 30)));
        assert!(!s.is_open(ist(20261108, 19, 0)));
        assert_eq!(s.session_close_ns(20261108), Some(ist(20261108, 19, 0)));
        // An empty override closes a weekday; a half day shortens one.
        assert!(!s.is_open(ist(20260126, 10, 0)));
        let half = s.with_session_override(20260115, vec![(9 * 60 + 15, 12 * 60)]).unwrap();
        assert!(half.is_open(ist(20260115, 11, 59)) && !half.is_open(ist(20260115, 12, 0)));
        assert_eq!(half.day_order_expiry_ns(ist(20260115, 10, 0)), Some(ist(20260115, 12, 0)));
        // Overrides are validated like windows.
        assert!(SessionSchedule::nse_equity()
            .with_session_override(20260115, vec![(900, 540)])
            .is_err());
        assert!(SessionSchedule::nse_equity().with_session_override(20260231, vec![]).is_err());
    }

    const DOC_Q4: &str = r#"{
        "name": "q4", "utc_offset": "+05:30", "valid_from": "2026-09-28", "valid_to": "2026-12-31",
        "sessions": {"NSE": {"open": "09:15", "close": "15:30"},
                     "NFO": {"open": "09:15", "close": "15:40"},
                     "MCX": {"windows": [["09:00", "17:00"], ["17:00", "23:55"]]}},
        "holidays": [{"date": "2026-10-02", "venues": ["NSE", "NFO", "MCX"]},
                     {"date": "2026-10-20", "venues": ["NSE"]}],
        "overrides": [{"venue": "MCX", "date": "2026-10-20", "open": "17:00", "close": "23:55"},
                      {"venue": "NSE", "date": "2026-11-08", "windows": [["18:15", "19:15"]]}]
    }"#;

    const DOC_DST: &str = r#"{
        "name": "q4-dst", "utc_offset": "+05:30", "valid_from": "2026-09-28", "valid_to": "2026-11-01",
        "sessions": {"MCX": {"windows": [["09:00", "17:00"], ["17:00", "23:30"]]}}
    }"#;

    #[test]
    fn a_calendar_document_gives_sessions_holidays_and_overrides() {
        let nse = SessionSchedule::from_calendars("NSE", &[DOC_Q4]).unwrap();
        assert!(nse.is_open(ist(20261015, 10, 0)));
        assert!(!nse.is_open(ist(20261002, 10, 0)), "holiday");
        assert!(!nse.is_open(ist(20261020, 10, 0)), "holiday for NSE");
        assert!(nse.is_open(ist(20261108, 18, 30)), "Muhurat on a Sunday");
        assert!(!nse.is_open(ist(20261108, 19, 15)));
        let nfo = SessionSchedule::from_calendars("NFO", &[DOC_Q4]).unwrap();
        assert!(nfo.is_open(ist(20261015, 15, 39)));
        assert!(nfo.is_open(ist(20261020, 10, 0)), "not a holiday for NFO");
        let mcx = SessionSchedule::from_calendars("MCX", &[DOC_Q4]).unwrap();
        assert!(!mcx.is_open(ist(20261020, 10, 0)), "the override opens only the evening");
        assert!(mcx.is_open(ist(20261020, 17, 0)) && mcx.is_open(ist(20261020, 23, 54)));
        assert!(!mcx.is_open(ist(20261002, 12, 0)));
        // A date no document covers is closed, not guessed.
        assert!(!nse.is_open(ist(20260915, 10, 0)));
        assert!(!nse.is_open(ist(20270105, 10, 0)));
    }

    #[test]
    fn the_latest_document_answers_for_the_dates_it_covers() {
        let mcx = SessionSchedule::from_calendars("MCX", &[DOC_Q4, DOC_DST]).unwrap();
        assert!(!mcx.is_open(ist(20261015, 23, 40)), "23:30 close in the DST weeks");
        assert!(mcx.is_open(ist(20261015, 23, 29)));
        assert!(mcx.is_open(ist(20261102, 23, 40)), "winter hours after the DST document ends");
        assert!(mcx.is_open(ist(20261120, 23, 54)));
    }

    #[test]
    fn a_bad_calendar_document_is_refused_with_its_name() {
        let msg =
            |docs: &[&str]| SessionSchedule::from_calendars("NSE", docs).unwrap_err().to_string();
        assert!(msg(&["{"]).contains("not JSON"));
        assert!(msg(&[r#"{"name":"x"}"#]).contains("utc_offset"));
        assert!(SessionSchedule::from_calendars("BSE", &[DOC_Q4])
            .unwrap_err()
            .to_string()
            .contains("no entry for venue \"BSE\""));
        let bad_time = DOC_Q4.replace("15:30", "25:30");
        assert!(msg(&[&bad_time]).contains("not an HH:MM time"));
        let bad_date = DOC_Q4.replace("2026-12-31", "2026-02-30");
        assert!(msg(&[&bad_date]).contains("valid_to"));
        let outside = DOC_Q4.replace("2026-10-02", "2027-10-02");
        assert!(msg(&[&outside]).contains("outside valid_from"));
        let other_zone = DOC_DST.replace("+05:30", "+00:00");
        assert!(SessionSchedule::from_calendars("MCX", &[DOC_Q4, &other_zone])
            .unwrap_err()
            .to_string()
            .contains("utc_offset differs"));
        assert!(SessionSchedule::from_calendars("NSE", &[]).is_err());
    }
}
