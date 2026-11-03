//! Reading a venue's schedule from dated calendar documents; the document
//! format is on [`SessionSchedule::from_calendars`].

use serde_json::Value;
use std::collections::BTreeSet;

use super::{day_from_yyyymmdd, Era, SessionSchedule, MINUTES_PER_DAY, NS_PER_MINUTE};
use crate::core::error::RaptorError;
use crate::core::Result;

fn refuse(doc: &str, what: impl std::fmt::Display) -> RaptorError {
    RaptorError::invalid_parameter(format!("calendar {doc}: {what}"))
}

/// `"YYYY-MM-DD"` as a `yyyymmdd` date.
fn date(doc: &str, v: Option<&Value>, key: &str) -> Result<i32> {
    let text = v
        .and_then(Value::as_str)
        .ok_or_else(|| refuse(doc, format!("{key} must be a YYYY-MM-DD string")))?;
    let b = text.as_bytes();
    let ok = b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && text.bytes().enumerate().all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit());
    let n = if ok { text.replace('-', "").parse::<i32>().ok() } else { None };
    match n {
        Some(n) if day_from_yyyymmdd(n).is_some() => Ok(n),
        _ => Err(refuse(doc, format!("{key} {text:?} is not a YYYY-MM-DD date"))),
    }
}

/// `"HH:MM"` as local minutes from midnight (`"24:00"` is the day's end).
fn minute(doc: &str, v: &Value, key: &str) -> Result<u32> {
    let text = v.as_str().ok_or_else(|| refuse(doc, format!("{key} must be an HH:MM string")))?;
    let parsed = text.split_once(':').and_then(|(h, m)| {
        (h.len() == 2 && m.len() == 2)
            .then(|| Some(h.parse::<u32>().ok()? * 60 + m.parse::<u32>().ok()?))
            .flatten()
    });
    match parsed {
        Some(n) if n <= MINUTES_PER_DAY && text.as_bytes()[3] < b'6' => Ok(n),
        _ => Err(refuse(doc, format!("{key} {text:?} is not an HH:MM time"))),
    }
}

/// `"+05:30"` as nanoseconds.
fn offset(doc: &str, v: Option<&Value>) -> Result<i64> {
    let text = v
        .and_then(Value::as_str)
        .ok_or_else(|| refuse(doc, "utc_offset is required, e.g. \"+05:30\""))?;
    let (sign, rest) = match text.as_bytes().first() {
        Some(b'+') => (1, &text[1..]),
        Some(b'-') => (-1, &text[1..]),
        _ => return Err(refuse(doc, format!("utc_offset {text:?} must be like \"+05:30\""))),
    };
    let m = minute(doc, &Value::String(rest.to_string()), "utc_offset")?;
    if m >= MINUTES_PER_DAY {
        return Err(refuse(doc, format!("utc_offset {text:?} must be within +-24 hours")));
    }
    Ok(sign * i64::from(m) * NS_PER_MINUTE)
}

fn windows_of(doc: &str, entry: &Value, what: &str) -> Result<Vec<(u32, u32)>> {
    if entry.get("closed").and_then(Value::as_bool) == Some(true) {
        return Ok(Vec::new());
    }
    if let Some(list) = entry.get("windows") {
        let list = list.as_array().ok_or_else(|| {
            refuse(doc, format!("{what}: windows must be a list of [open, close]"))
        })?;
        return list
            .iter()
            .map(|w| match w.as_array().map(Vec::as_slice) {
                Some([o, c]) => Ok((minute(doc, o, what)?, minute(doc, c, what)?)),
                _ => Err(refuse(doc, format!("{what}: a window is [\"HH:MM\", \"HH:MM\"]"))),
            })
            .collect();
    }
    match (entry.get("open"), entry.get("close")) {
        (Some(o), Some(c)) => Ok(vec![(minute(doc, o, what)?, minute(doc, c, what)?)]),
        _ => Err(refuse(doc, format!("{what}: needs open and close, windows, or closed"))),
    }
}

fn weekdays_of(doc: &str, entry: &Value) -> Result<Vec<usize>> {
    const NAMES: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
    let Some(list) = entry.get("weekdays") else { return Ok(vec![0, 1, 2, 3, 4]) };
    list.as_array()
        .ok_or_else(|| refuse(doc, "weekdays must be a list of mon..sun"))?
        .iter()
        .map(|d| {
            d.as_str()
                .and_then(|n| NAMES.iter().position(|w| n.eq_ignore_ascii_case(w)))
                .ok_or_else(|| refuse(doc, format!("weekday {d} is not one of mon..sun")))
        })
        .collect()
}

fn names<'a>(entry: &'a Value, plural: &str, singular: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    for key in [plural, singular] {
        match entry.get(key) {
            Some(Value::String(s)) => out.push(s.as_str()),
            Some(Value::Array(a)) => out.extend(a.iter().filter_map(Value::as_str)),
            _ => {}
        }
    }
    out
}

pub(super) fn from_calendars(venue: &str, documents: &[&str]) -> Result<SessionSchedule> {
    if documents.is_empty() {
        return Err(RaptorError::invalid_parameter("from_calendars needs at least one document"));
    }
    let mut schedule: Option<SessionSchedule> = None;
    let mut tz: Option<i64> = None;
    let mut holidays = BTreeSet::new();
    let mut covered = Vec::new();
    let mut eras = Vec::new();
    let mut overrides = std::collections::BTreeMap::new();

    for (n, text) in documents.iter().enumerate() {
        let doc: Value = serde_json::from_str(text)
            .map_err(|e| RaptorError::invalid_parameter(format!("calendar #{n}: not JSON: {e}")))?;
        let name =
            doc.get("name").and_then(Value::as_str).map_or_else(|| format!("#{n}"), str::to_string);
        let name = name.as_str();
        let this_tz = offset(name, doc.get("utc_offset"))?;
        if *tz.get_or_insert(this_tz) != this_tz {
            return Err(refuse(name, "utc_offset differs from the first document's"));
        }
        let from = date(name, doc.get("valid_from"), "valid_from")?;
        let to = date(name, doc.get("valid_to"), "valid_to")?;
        let (from_day, to_day) = (day_from_yyyymmdd(from).unwrap(), day_from_yyyymmdd(to).unwrap());
        if to_day < from_day {
            return Err(refuse(name, "valid_to is before valid_from"));
        }
        let entry = doc
            .get("sessions")
            .and_then(|s| s.get(venue))
            .ok_or_else(|| refuse(name, format!("sessions has no entry for venue {venue:?}")))?;
        let day = windows_of(name, entry, "sessions")?;
        let mut week: [Vec<(u32, u32)>; 7] = Default::default();
        for wd in weekdays_of(name, entry)? {
            week[wd] = day.clone();
        }
        SessionSchedule::check_windows(&mut week).map_err(|e| refuse(name, e))?;
        if schedule.is_none() {
            schedule = Some(
                SessionSchedule::new(this_tz, week.clone(), [], None)
                    .map_err(|e| refuse(name, e))?,
            );
        }
        eras.push(Era { from: from_day, to: to_day, windows: week });
        covered.push((from_day, to_day));

        for h in doc.get("holidays").and_then(Value::as_array).into_iter().flatten() {
            let listed =
                names(h, "venues", "venue").into_iter().chain(names(h, "exchanges", "exchange"));
            if listed.into_iter().any(|v| v == venue) {
                let d = date(name, h.get("date"), "holiday date")?;
                if !(from..=to).contains(&d) {
                    return Err(refuse(
                        name,
                        format!("holiday {d} is outside valid_from..valid_to"),
                    ));
                }
                holidays.insert(d);
                overrides.remove(&d);
            }
        }
        for o in doc.get("overrides").and_then(Value::as_array).into_iter().flatten() {
            let listed =
                names(o, "venues", "venue").into_iter().chain(names(o, "exchanges", "exchange"));
            if listed.into_iter().any(|v| v == venue) {
                let d = date(name, o.get("date"), "override date")?;
                if !(from..=to).contains(&d) {
                    return Err(refuse(
                        name,
                        format!("override {d} is outside valid_from..valid_to"),
                    ));
                }
                let mut w = windows_of(name, o, "override")?;
                SessionSchedule::check_day(&d.to_string(), &mut w).map_err(|e| refuse(name, e))?;
                holidays.remove(&d);
                overrides.insert(d, w);
            }
        }
    }
    let mut out = schedule.expect("at least one document");
    out.eras = eras;
    out.covered = covered;
    out.holidays = holidays;
    out.overrides = overrides;
    Ok(out)
}
