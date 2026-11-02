//! Standalone resampling on the time-window builder: ticks to bars, and bars to coarser
//! bars on a grid you declare.
//!
//! This is [`TimeBarBuilder`] with what a feed that runs for days needs and a batch
//! `[start, end)` aggregation does not:
//!
//! - **Which edge a boundary print belongs to** ([`Closed`]). `Left` is the builder's own
//!   `[start, end)`; `Right` is `(start, end]`, where a print exactly on a boundary
//!   belongs to the bucket it closes and a bar stamped at its close lands in the bucket
//!   that close ends.
//! - **A grid you place.** `anchor_ns` puts the boundaries at `anchor + k * width` on the
//!   local clock `tz_offset_ns` ahead of UTC (a 30-minute grid from a 09:15 open), or
//!   `session_anchored` restarts the grid at every session open you report.
//! - **Closing by the clock** ([`CloseOn`]): a bucket closes as soon as time reaches its
//!   end (`on_clock`), or only when the next print arrives.
//! - **Empty intervals** ([`Gaps`]): skipped, or carried as a flat zero-volume bar at the
//!   last close, so a quiet minute is still a bar.
//! - **Partial buckets**: every bar says how many source records built it (`n_source`),
//!   whether a resampled bar saw all the source bars its width spans (`complete`), and what
//!   closed it ([`ClosedBy`]). `min_source` drops a bucket built from too few.
//! - **Refusals, not repairs**: a record earlier than the open bucket, a non-finite price,
//!   a source bar off the source grid or of another width, and a width that is not a
//!   multiple of the source width are errors. Nothing is merged into the wrong bar.
//!
//! A bar is stamped with its bucket's end, as everywhere in the engine, so a bar stamped
//! `t` holds nothing at or after `t` (for `Left`) and nothing after `t` (for `Right`).

use crate::core::types::{OhlcvBar, Timestamp};
use crate::data::aggregate::{Partial, SourceRecord, TimeBarBuilder};

/// Which edge of a bucket a record exactly on a boundary belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    /// `[start, end)`: the boundary print opens the next bucket.
    Left,
    /// `(start, end]`: the boundary print closes the bucket it ends.
    Right,
}

impl Closed {
    pub fn as_str(self) -> &'static str {
        match self {
            Closed::Left => "left",
            Closed::Right => "right",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ResampleError> {
        match s {
            "left" => Ok(Closed::Left),
            "right" => Ok(Closed::Right),
            _ => Err(ResampleError::Invalid("closed is 'left' or 'right'".into())),
        }
    }
}

/// What an empty interval of the grid emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gaps {
    Skip,
    /// A flat bar at the last close with zero volume.
    Carry,
}

impl Gaps {
    pub fn as_str(self) -> &'static str {
        match self {
            Gaps::Skip => "skip",
            Gaps::Carry => "carry",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ResampleError> {
        match s {
            "skip" => Ok(Gaps::Skip),
            "carry" => Ok(Gaps::Carry),
            _ => Err(ResampleError::Invalid("gaps is 'skip' or 'carry'".into())),
        }
    }
}

/// When the clock may close a bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseOn {
    /// As soon as time reaches the bucket's end.
    Clock,
    /// Only when the next record of the source arrives.
    NextPrint,
}

impl CloseOn {
    pub fn as_str(self) -> &'static str {
        match self {
            CloseOn::Clock => "clock",
            CloseOn::NextPrint => "next_print",
        }
    }

    pub fn parse(s: &str) -> Result<Self, ResampleError> {
        match s {
            "clock" => Ok(CloseOn::Clock),
            "next_print" => Ok(CloseOn::NextPrint),
            _ => Err(ResampleError::Invalid("close_on is 'clock' or 'next_print'".into())),
        }
    }
}

/// What closed a bar, so every bar is auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosedBy {
    /// The clock reached the bucket's end.
    Clock,
    /// A later record arrived, or the last source bar of the bucket did.
    Print,
    /// A flat bar for an empty interval.
    Carry,
    /// A session open or close.
    Session,
    /// The end of the stream.
    Flush,
}

impl ClosedBy {
    pub fn as_str(self) -> &'static str {
        match self {
            ClosedBy::Clock => "clock",
            ClosedBy::Print => "print",
            ClosedBy::Carry => "carry",
            ClosedBy::Session => "session",
            ClosedBy::Flush => "flush",
        }
    }
}

/// How a resampler is declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleOptions {
    /// Bucket width in ns.
    pub width_ns: i64,
    /// Where the grid sits: boundaries at `anchor_ns + k * width_ns` on the local clock.
    pub anchor_ns: i64,
    /// The local clock's offset from UTC (IST is [`crate::data::IST_OFFSET_NS`]).
    pub tz_offset_ns: i64,
    pub closed: Closed,
    pub gaps: Gaps,
    pub close_on: CloseOn,
    /// Width of the source bars (0 for trades). Bars of another width are refused, the
    /// width must be a multiple of it, and the anchor must sit on its grid.
    pub source_ns: i64,
    /// Drop a bucket built from fewer source records (0: none is dropped).
    pub min_source: u64,
    /// Restart the grid at every session open reported with [`TimeBarBuilder::session`];
    /// records before the first open are not bucketed. `anchor_ns` and `tz_offset_ns` are
    /// then unused.
    pub session_anchored: bool,
}

impl ResampleOptions {
    /// `width_ns` buckets on the epoch grid, left-closed, skipping gaps, closing on the
    /// next record: what [`TimeBarBuilder::new`] does, plus the refusals.
    pub fn new(width_ns: i64) -> Self {
        Self {
            width_ns,
            anchor_ns: 0,
            tz_offset_ns: 0,
            closed: Closed::Left,
            gaps: Gaps::Skip,
            close_on: CloseOn::NextPrint,
            source_ns: 0,
            min_source: 0,
            session_anchored: false,
        }
    }
}

/// Why a resampler refused a call or a declaration.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ResampleError {
    #[error("{0}")]
    Invalid(String),
    #[error("record at {ts} is before the open bucket ending {end}")]
    Late { ts: Timestamp, end: Timestamp },
    #[error("{0}")]
    NotFinite(&'static str),
    #[error("a bar of width {got} on a source of width {want}")]
    PeriodMismatch { got: i64, want: i64 },
    #[error("bar at {0} is off the source grid")]
    OffGrid(Timestamp),
    #[error("this resampler takes {0}")]
    WrongInput(&'static str),
    #[error("timestamp {0} is outside the representable grid")]
    OutOfRange(Timestamp),
}

/// A closed bar and how it came to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResampledBar {
    /// The bucket's end, or the session event that closed it early.
    pub bar: OhlcvBar,
    /// Source records (prints or bars) the bar was built from; 0 for a carried bar.
    pub n_source: u64,
    /// False for a bar of resampled bars built from fewer source bars than its width spans.
    pub complete: bool,
    pub closed_by: ClosedBy,
}

/// What the resampler API keeps beside the builder's window and partial bar.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResampleState {
    opts: ResampleOptions,
    /// The grid is `align_offset_ns` until a session-anchored one is told its first open.
    grid_set: bool,
    count: u64,
    /// Window of the last bucket closed or carried, so empty intervals after it are filled.
    last_window: Option<i64>,
    last_close: f64,
}

impl ResampleState {
    pub(crate) fn new(width_ns: i64) -> Self {
        Self {
            opts: ResampleOptions::new(width_ns),
            grid_set: true,
            count: 0,
            last_window: None,
            last_close: 0.0,
        }
    }
}

impl TimeBarBuilder {
    /// A resampler declared by `opts`.
    pub fn resampler(opts: ResampleOptions) -> Result<Self, ResampleError> {
        let w = opts.width_ns;
        if w <= 0 {
            return Err(ResampleError::Invalid("the width must be positive".into()));
        }
        let off = if opts.session_anchored {
            0
        } else {
            let anchor = i128::from(opts.anchor_ns);
            // boundaries at anchor + k * width on the local clock: (ts + tz - anchor) % w == 0
            let off = i128::from(opts.tz_offset_ns) - anchor;
            i64::try_from(off.rem_euclid(i128::from(w)))
                .map_err(|_| ResampleError::Invalid("the anchor is out of range".into()))?
        };
        if opts.source_ns < 0 {
            return Err(ResampleError::Invalid("the source width cannot be negative".into()));
        }
        if opts.source_ns > 0 {
            if w % opts.source_ns != 0 {
                return Err(ResampleError::Invalid(format!(
                    "{w} is not a multiple of the source width {}",
                    opts.source_ns
                )));
            }
            if !opts.session_anchored && off.rem_euclid(opts.source_ns) != 0 {
                return Err(ResampleError::Invalid(
                    "the anchor must sit on the source grid".into(),
                ));
            }
            let full = (w / opts.source_ns) as u64;
            if opts.min_source > full {
                return Err(ResampleError::Invalid(format!(
                    "min_source {} is more than the {full} source bars a bucket spans",
                    opts.min_source
                )));
            }
        }
        let mut b = TimeBarBuilder::new(w, off);
        b.resample = ResampleState {
            opts,
            grid_set: !opts.session_anchored,
            count: 0,
            last_window: None,
            last_close: 0.0,
        };
        Ok(b)
    }

    fn state(&self) -> &ResampleState {
        &self.resample
    }

    fn state_mut(&mut self) -> &mut ResampleState {
        &mut self.resample
    }

    /// The options this resampler was declared with.
    pub fn options(&self) -> ResampleOptions {
        self.state().opts
    }

    /// Window index of `ts`; `None` before a session-anchored grid has its first open.
    fn rwindow(&self, ts: Timestamp) -> Result<Option<i64>, ResampleError> {
        let st = self.state();
        if !st.grid_set {
            return Ok(None);
        }
        let shift = i128::from(st.opts.closed == Closed::Right);
        let x = i128::from(ts) + i128::from(self.align_offset_ns) - shift;
        i64::try_from(x.div_euclid(i128::from(self.width_ns)))
            .map(Some)
            .map_err(|_| ResampleError::OutOfRange(ts))
    }

    /// The end (exclusive for `Left`, inclusive for `Right`) of a window, in the source clock.
    fn rend(&self, window: i64) -> Result<Timestamp, ResampleError> {
        let end =
            (i128::from(window) + 1) * i128::from(self.width_ns) - i128::from(self.align_offset_ns);
        Timestamp::try_from(end).map_err(|_| ResampleError::OutOfRange(window))
    }

    fn full_count(&self) -> u64 {
        let st = self.state();
        if st.opts.source_ns > 0 {
            (self.width_ns / st.opts.source_ns) as u64
        } else {
            0
        }
    }

    /// Close the open bucket, stamped `stamp`, unless it holds too few source records.
    fn close_bucket(&mut self, stamp: Timestamp, by: ClosedBy, out: &mut Vec<ResampledBar>) {
        let (window, partial) = match (self.window.take(), self.partial.take()) {
            (Some(w), Some(p)) => (w, p),
            _ => return,
        };
        let full = self.full_count();
        let st = self.state_mut();
        let count = std::mem::take(&mut st.count);
        st.last_window = Some(window);
        if count < st.opts.min_source {
            return;
        }
        st.last_close = partial.close;
        let complete = st.opts.source_ns == 0 || count >= full;
        out.push(ResampledBar {
            bar: partial.into_bar(stamp),
            n_source: count,
            complete,
            closed_by: by,
        });
    }

    /// Flat bars for every empty window after the last bucket up to, not including, `to`.
    fn carry_to(&mut self, to: i64, out: &mut Vec<ResampledBar>) -> Result<(), ResampleError> {
        let st = self.state();
        if st.opts.gaps != Gaps::Carry {
            return Ok(());
        }
        let (Some(last), close) = (st.last_window, st.last_close) else {
            return Ok(());
        };
        let mut w = last;
        while w + 1 < to {
            w += 1;
            let ts = self.rend(w)?;
            out.push(ResampledBar {
                bar: OhlcvBar {
                    timestamp: ts,
                    open: close,
                    high: close,
                    low: close,
                    close,
                    volume: 0.0,
                },
                n_source: 0,
                complete: true,
                closed_by: ClosedBy::Carry,
            });
        }
        self.state_mut().last_window = Some(w.max(last));
        Ok(())
    }

    /// Take one record. Closes the open bucket when the record is in a later one (by the
    /// clock under [`CloseOn::Clock`], by this record otherwise) and fills the empty
    /// intervals between. A record earlier than the open bucket is refused, and nothing is
    /// emitted or changed.
    pub fn absorb(
        &mut self,
        rec: &SourceRecord,
        out: &mut Vec<ResampledBar>,
    ) -> Result<(), ResampleError> {
        let prices = [rec.open, rec.high, rec.low, rec.close];
        if prices.iter().any(|p| !p.is_finite()) || !rec.volume.is_finite() || rec.volume < 0.0 {
            return Err(ResampleError::NotFinite(
                "a record needs finite prices and a finite, non-negative volume",
            ));
        }
        let Some(win) = self.rwindow(rec.timestamp)? else {
            return Ok(()); // a session-anchored grid with no open yet
        };
        // the open bucket, or the last one closed (by the clock) when none is open
        let behind = self.window.or(self.state().last_window.filter(|_| self.window.is_none()));
        if let Some(prev) = behind {
            let too_early = if self.window.is_some() { win < prev } else { win <= prev };
            if too_early {
                return Err(ResampleError::Late { ts: rec.timestamp, end: self.rend(prev)? });
            }
        }
        // everything that can fail is computed before the first emit
        let cur_end = self.window.map(|c| self.rend(c)).transpose()?;
        if let (Some(cur), Some(end)) = (self.window, cur_end) {
            if win > cur {
                // a trade past the boundary is the clock's signal under `Clock`; a source
                // bar past it is just the next record
                let o = self.state().opts;
                let by = if o.close_on == CloseOn::Clock && o.source_ns == 0 {
                    ClosedBy::Clock
                } else {
                    ClosedBy::Print
                };
                self.close_bucket(end, by, out);
            }
        }
        if self.window.is_none() {
            self.carry_to(win, out)?;
            self.window = Some(win);
            self.partial = Some(Partial::start(rec));
            self.state_mut().count = 1;
        } else if let Some(p) = self.partial.as_mut() {
            p.absorb(rec);
            self.state_mut().count += 1;
        }
        Ok(())
    }

    /// One trade, then the clock at its instant.
    pub fn push_trade(
        &mut self,
        ts: Timestamp,
        price: f64,
        qty: f64,
        out: &mut Vec<ResampledBar>,
    ) -> Result<(), ResampleError> {
        if self.state().opts.source_ns > 0 {
            return Err(ResampleError::WrongInput("bars, not trades"));
        }
        self.absorb(&SourceRecord::trade(ts, price, qty), out)?;
        self.on_clock(ts, out)
    }

    /// One source bar of width `period_ns` (0: not stated), then the clock at its closing
    /// instant. A bar must sit on the source grid; the bucket's last source bar closes it
    /// at once.
    #[allow(clippy::too_many_arguments)]
    pub fn push_bar(
        &mut self,
        ts: Timestamp,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        period_ns: i64,
        out: &mut Vec<ResampledBar>,
    ) -> Result<(), ResampleError> {
        let (source, closed) = (self.state().opts.source_ns, self.state().opts.closed);
        if source == 0 {
            return Err(ResampleError::WrongInput("trades, not bars (declare source_ns)"));
        }
        if period_ns != 0 && period_ns != source {
            return Err(ResampleError::PeriodMismatch { got: period_ns, want: source });
        }
        if self.state().grid_set
            && (i128::from(ts) + i128::from(self.align_offset_ns)).rem_euclid(i128::from(source))
                != 0
        {
            return Err(ResampleError::OffGrid(ts));
        }
        let rec =
            SourceRecord { timestamp: ts, open, high, low, close, volume, signed_volume: 0.0 };
        self.absorb(&rec, out)?;
        // the bar's own closing instant: its stamp if stamped at the close, else a width on
        let closing = match closed {
            Closed::Right => i128::from(ts),
            Closed::Left => i128::from(ts) + i128::from(source),
        };
        if let Some(cur) = self.window {
            if i128::from(self.rend(cur)?) == closing {
                let end = self.rend(cur)?;
                self.close_bucket(end, ClosedBy::Print, out);
            }
        }
        self.on_clock(ts, out)
    }

    /// Time has reached `ts`: under [`CloseOn::Clock`], close the open bucket if its end has
    /// passed and carry the empty intervals up to `ts`. Nothing under
    /// [`CloseOn::NextPrint`].
    pub fn on_clock(
        &mut self,
        ts: Timestamp,
        out: &mut Vec<ResampledBar>,
    ) -> Result<(), ResampleError> {
        if self.state().opts.close_on != CloseOn::Clock {
            return Ok(());
        }
        let Some(now) = self.rwindow(ts)? else {
            return Ok(());
        };
        if let Some(cur) = self.window {
            let end = self.rend(cur)?;
            if ts < end {
                return Ok(());
            }
            self.close_bucket(end, ClosedBy::Clock, out);
        }
        self.carry_to(now, out)
    }

    /// A session event. Under a session-anchored grid an open closes what is open at `ts` and
    /// restarts the grid there; a close stamps the session's last bucket at `ts`. A grid
    /// that is not session-anchored only moves the clock.
    pub fn session(
        &mut self,
        open: bool,
        ts: Timestamp,
        out: &mut Vec<ResampledBar>,
    ) -> Result<(), ResampleError> {
        if !self.state().opts.session_anchored {
            return self.on_clock(ts, out);
        }
        if !open {
            if let Some(cur) = self.window {
                if self.rend(cur)? >= ts {
                    self.close_bucket(ts, ClosedBy::Session, out);
                }
            }
            return self.on_clock(ts, out);
        }
        self.close_bucket(ts, ClosedBy::Session, out);
        let w = self.width_ns;
        self.align_offset_ns = i64::try_from((-i128::from(ts)).rem_euclid(i128::from(w)))
            .map_err(|_| ResampleError::OutOfRange(ts))?;
        let st = self.state_mut();
        st.grid_set = true;
        st.last_window = None;
        Ok(())
    }

    /// The instant the clock would next close the open bucket, if any.
    pub fn next_close(&self) -> Option<Timestamp> {
        if self.state().opts.close_on != CloseOn::Clock {
            return None;
        }
        self.window.and_then(|w| self.rend(w).ok())
    }

    /// The open bucket as it stands, for a stream that has truly ended: stamped at its end.
    pub fn flush_resampled(&mut self, out: &mut Vec<ResampledBar>) {
        if let Some(cur) = self.window {
            if let Ok(end) = self.rend(cur) {
                self.close_bucket(end, ClosedBy::Flush, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = 1_000_000_000;
    const M: i64 = 60 * S;

    fn run(opts: ResampleOptions, events: &[(i64, f64)]) -> Vec<ResampledBar> {
        let mut b = TimeBarBuilder::resampler(opts).unwrap();
        let mut out = Vec::new();
        for &(ts, px) in events {
            b.push_trade(ts, px, 1.0, &mut out).unwrap();
        }
        b.flush_resampled(&mut out);
        out
    }

    #[test]
    fn left_and_right_put_a_boundary_print_in_different_bars() {
        let ev = [(M / 2, 100.0), (M, 101.0), (M + 1, 102.0)];
        let left = run(ResampleOptions::new(M), &ev);
        let right = run(ResampleOptions { closed: Closed::Right, ..ResampleOptions::new(M) }, &ev);
        // left: the print at 60s opens the second bar
        assert_eq!(left.iter().map(|b| b.bar.timestamp).collect::<Vec<_>>(), [M, 2 * M]);
        assert_eq!(left[0].bar.close, 100.0);
        // right: the print at 60s closes the first bar
        assert_eq!(right[0].bar.close, 101.0);
        assert_eq!(right[0].bar.volume, 2.0);
        assert_eq!(right[1].bar.open, 102.0);
    }

    #[test]
    fn the_default_resampler_agrees_with_the_bar_builder() {
        use crate::data::BarBuilder;
        let mut legacy = TimeBarBuilder::new(M, 0);
        let mut legacy_bars = Vec::new();
        let mut events = Vec::new();
        for i in 0..500i64 {
            let ts = i * 7 * S + (i % 5) * 1_000;
            let px = 100.0 + ((i * 37) % 11) as f64;
            events.push((ts, px));
            legacy_bars.extend(legacy.push(&SourceRecord::trade(ts, px, 1.0)));
        }
        legacy_bars.extend(legacy.flush());
        let got = run(ResampleOptions::new(M), &events);
        assert_eq!(got.len(), legacy_bars.len());
        for (g, l) in got.iter().zip(&legacy_bars) {
            assert_eq!(g.bar, *l);
        }
    }

    #[test]
    fn the_clock_closes_a_bucket_without_waiting_for_the_next_print() {
        let opts = ResampleOptions {
            closed: Closed::Right,
            close_on: CloseOn::Clock,
            ..ResampleOptions::new(M)
        };
        let mut b = TimeBarBuilder::resampler(opts).unwrap();
        let mut out = Vec::new();
        b.push_trade(10 * S, 100.0, 1.0, &mut out).unwrap();
        assert!(out.is_empty());
        assert_eq!(b.next_close(), Some(M));
        b.on_clock(59 * S, &mut out).unwrap();
        assert!(out.is_empty());
        b.on_clock(M, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].bar.timestamp, out[0].closed_by), (M, ClosedBy::Clock));
        // a print exactly on the boundary closes the bucket it ends, at once
        let mut b = TimeBarBuilder::resampler(opts).unwrap();
        let mut out = Vec::new();
        b.push_trade(10 * S, 100.0, 1.0, &mut out).unwrap();
        b.push_trade(M, 105.0, 1.0, &mut out).unwrap();
        assert_eq!((out[0].bar.close, out[0].bar.volume), (105.0, 2.0));
    }

    #[test]
    fn carried_gaps_fill_every_empty_interval() {
        let opts = ResampleOptions { gaps: Gaps::Carry, ..ResampleOptions::new(M) };
        let got = run(opts, &[(S, 100.0), (4 * M + S, 110.0)]);
        let stamps: Vec<_> =
            got.iter().map(|b| (b.bar.timestamp, b.closed_by, b.bar.volume)).collect();
        assert_eq!(
            stamps,
            [
                (M, ClosedBy::Print, 1.0),
                (2 * M, ClosedBy::Carry, 0.0),
                (3 * M, ClosedBy::Carry, 0.0),
                (4 * M, ClosedBy::Carry, 0.0),
                (5 * M, ClosedBy::Flush, 1.0)
            ]
        );
        assert!(got[1..4].iter().all(|b| b.bar.close == 100.0 && b.n_source == 0));
    }

    #[test]
    fn carry_survives_a_clock_close_before_the_next_print() {
        // the clock closes the first bucket; the next print comes minutes later
        let opts = ResampleOptions {
            gaps: Gaps::Carry,
            close_on: CloseOn::Clock,
            closed: Closed::Right,
            ..ResampleOptions::new(M)
        };
        let mut b = TimeBarBuilder::resampler(opts).unwrap();
        let mut out = Vec::new();
        b.push_trade(S, 100.0, 1.0, &mut out).unwrap();
        b.on_clock(M, &mut out).unwrap();
        b.push_trade(5 * M - S, 90.0, 1.0, &mut out).unwrap();
        let stamps: Vec<_> = out.iter().map(|x| (x.bar.timestamp, x.closed_by)).collect();
        assert_eq!(
            stamps,
            [
                (M, ClosedBy::Clock),
                (2 * M, ClosedBy::Carry),
                (3 * M, ClosedBy::Carry),
                (4 * M, ClosedBy::Carry)
            ]
        );
    }

    #[test]
    fn an_anchor_and_a_zone_place_the_grid() {
        // 30-minute bars from a 09:15 IST open: boundaries at 09:15, 09:45, ... local
        let open = (9 * 60 + 15) * M;
        let opts = ResampleOptions {
            anchor_ns: open,
            tz_offset_ns: crate::data::IST_OFFSET_NS,
            closed: Closed::Right,
            ..ResampleOptions::new(30 * M)
        };
        let day = 20_000 * 86_400 * S;
        let at = |h: i64, m: i64| day + (h * 60 + m) * M - crate::data::IST_OFFSET_NS;
        let got = run(
            opts,
            &[(at(9, 16), 100.0), (at(9, 44), 101.0), (at(9, 45), 102.0), (at(9, 46), 103.0)],
        );
        assert_eq!(got[0].bar.timestamp, at(9, 45));
        assert_eq!(got[0].bar.close, 102.0);
        assert_eq!(got[1].bar.open, 103.0);
    }

    #[test]
    fn a_resampled_bar_says_whether_it_saw_every_source_bar() {
        let opts =
            ResampleOptions { closed: Closed::Right, source_ns: M, ..ResampleOptions::new(5 * M) };
        let mut b = TimeBarBuilder::resampler(opts).unwrap();
        let mut out = Vec::new();
        // minutes 1..=5 of the first bucket, then only minute 6 and 7 of the next
        for i in 1..=5 {
            b.push_bar(i * M, 100.0, 101.0 + i as f64, 99.0, 100.0 + i as f64, 10.0, M, &mut out)
                .unwrap();
        }
        // the fifth source bar closed the bucket at once
        assert_eq!(out.len(), 1);
        assert!(out[0].complete && out[0].n_source == 5 && out[0].closed_by == ClosedBy::Print);
        assert_eq!((out[0].bar.high, out[0].bar.volume), (106.0, 50.0));
        for i in 6..=7 {
            b.push_bar(i * M, 100.0, 100.0, 100.0, 100.0, 1.0, M, &mut out).unwrap();
        }
        b.flush_resampled(&mut out);
        let last = out.last().unwrap();
        assert!(!last.complete && last.n_source == 2 && last.closed_by == ClosedBy::Flush);
        // and min_source drops it
        let dropping = ResampleOptions { min_source: 3, ..opts };
        let mut b = TimeBarBuilder::resampler(dropping).unwrap();
        let mut out = Vec::new();
        for i in 6..=7 {
            b.push_bar(i * M, 100.0, 100.0, 100.0, 100.0, 1.0, M, &mut out).unwrap();
        }
        b.flush_resampled(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn late_and_malformed_input_is_refused_and_changes_nothing() {
        let mut b = TimeBarBuilder::resampler(ResampleOptions::new(M)).unwrap();
        let mut out = Vec::new();
        b.push_trade(3 * M, 100.0, 1.0, &mut out).unwrap();
        assert!(matches!(b.push_trade(M, 99.0, 1.0, &mut out), Err(ResampleError::Late { .. })));
        assert!(matches!(
            b.push_trade(4 * M, f64::NAN, 1.0, &mut out),
            Err(ResampleError::NotFinite(_))
        ));
        assert!(matches!(
            b.push_trade(4 * M, 1.0, -1.0, &mut out),
            Err(ResampleError::NotFinite(_))
        ));
        assert!(out.is_empty());
        b.flush_resampled(&mut out);
        assert_eq!((out.len(), out[0].n_source, out[0].bar.close), (1, 1, 100.0));

        let bars =
            ResampleOptions { source_ns: M, closed: Closed::Right, ..ResampleOptions::new(5 * M) };
        let mut b = TimeBarBuilder::resampler(bars).unwrap();
        assert!(matches!(
            b.push_bar(M, 1.0, 1.0, 1.0, 1.0, 1.0, 2 * M, &mut out),
            Err(ResampleError::PeriodMismatch { .. })
        ));
        assert!(matches!(
            b.push_bar(M + S, 1.0, 1.0, 1.0, 1.0, 1.0, M, &mut out),
            Err(ResampleError::OffGrid(_))
        ));
        assert!(matches!(b.push_trade(M, 1.0, 1.0, &mut out), Err(ResampleError::WrongInput(_))));
        // declarations that approximate are refused
        assert!(TimeBarBuilder::resampler(ResampleOptions {
            source_ns: 2 * M,
            ..ResampleOptions::new(5 * M)
        })
        .is_err());
        assert!(TimeBarBuilder::resampler(ResampleOptions {
            source_ns: M,
            anchor_ns: S,
            ..ResampleOptions::new(5 * M)
        })
        .is_err());
        assert!(TimeBarBuilder::resampler(ResampleOptions::new(0)).is_err());
    }

    #[test]
    fn a_session_anchored_grid_restarts_at_each_open() {
        let opts = ResampleOptions {
            session_anchored: true,
            closed: Closed::Right,
            close_on: CloseOn::Clock,
            ..ResampleOptions::new(30 * M)
        };
        let mut b = TimeBarBuilder::resampler(opts).unwrap();
        let mut out = Vec::new();
        // a print before the first open is not bucketed
        b.push_trade(5 * M, 99.0, 1.0, &mut out).unwrap();
        let open = 100 * M + 17 * S;
        b.session(true, open, &mut out).unwrap();
        b.push_trade(open + M, 100.0, 1.0, &mut out).unwrap();
        b.push_trade(open + 31 * M, 101.0, 1.0, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].bar.timestamp, open + 30 * M);
        // the session closes mid-bucket: the last bucket is stamped at the close
        b.session(false, open + 45 * M, &mut out).unwrap();
        assert_eq!((out[1].bar.timestamp, out[1].closed_by), (open + 45 * M, ClosedBy::Session));
        // the next open restarts the grid on its own instant
        let next = open + 24 * 60 * M;
        b.session(true, next, &mut out).unwrap();
        b.push_trade(next + 2 * M, 50.0, 1.0, &mut out).unwrap();
        b.flush_resampled(&mut out);
        assert_eq!(out.last().unwrap().bar.timestamp, next + 30 * M);
    }
}
