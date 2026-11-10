//! A run fed in windows: the batch tape cut into consecutive pieces, each read with a
//! [`MergeCursor`] of its own, delivering the events the whole tape's cursor would.
//!
//! [`RuntimeCore::advance_window`] opens a window over a [`MarketStore`] holding only that window's
//! rows and returns a [`WindowCursor`]; [`RuntimeCore::end_window`] closes it. What persists
//! from window to window is in the core (books, indicator bank, aggregators) and in
//! [`WindowBase`]: how many rows of each kind and how many events each instrument has had, so
//! that the rows and ordinals a window's events carry (`EventRef::row`, `EventRef::local_idx`,
//! hence `ctx.idx`, `entry_idx` and `exit_idx`) are whole-run numbers, not window-local ones.
//!
//! # What a window may be
//!
//! A timestamp group (the events sharing a timestamp and a kind, see
//! `docs/architecture/event-semantics.md`) is applied before any of it is decided, so a window
//! may not cut one: every event of a window is stamped strictly after the last event of the
//! window before it. A window that starts at or before that time is refused (`window_overlap`),
//! and nothing changes. Anything else is a valid cut, including one between two bars of the
//! same instrument at different times, one between an instrument's trades and its next bar,
//! and an empty window.
//!
//! # Rows are three counts
//!
//! An instrument has a row count per kind (bar rows, tick rows, depth rows), and a separate
//! event count, because one tick row can be a trade and a quote, and a row can be neither. The
//! caller does not pass them: the core adds up what each window held, so a cut cannot be
//! misstated.

use crate::core::error::RaptorError;
use crate::core::Result;
use crate::runtime::cursor::{EventKind, EventRef, MergeCursor};
use crate::runtime::ids::InstrumentId;
use crate::runtime::runner::{Pushed, RuntimeCore};
use crate::runtime::store::MarketStore;

/// What the windows before this one delivered, per instrument.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WindowBase {
    bars: Vec<u64>,
    ticks: Vec<u64>,
    depth: Vec<u64>,
    events: Vec<u64>,
    last_ts: Option<i64>,
    windows: u64,
}

impl WindowBase {
    fn grow_to(&mut self, instruments: usize) {
        for v in [&mut self.bars, &mut self.ticks, &mut self.depth, &mut self.events] {
            if v.len() < instruments {
                v.resize(instruments, 0);
            }
        }
    }

    /// Bar rows of `instrument` that earlier windows held.
    pub fn bar_rows(&self, instrument: InstrumentId) -> u64 {
        self.bars.get(instrument.index()).copied().unwrap_or(0)
    }

    /// Tick rows (every row, whether or not it yielded an event) earlier windows held.
    pub fn tick_rows(&self, instrument: InstrumentId) -> u64 {
        self.ticks.get(instrument.index()).copied().unwrap_or(0)
    }

    /// Depth snapshots earlier windows held.
    pub fn depth_rows(&self, instrument: InstrumentId) -> u64 {
        self.depth.get(instrument.index()).copied().unwrap_or(0)
    }

    /// Events earlier windows delivered for `instrument`: the next event's ordinal.
    pub fn events(&self, instrument: InstrumentId) -> u64 {
        self.events.get(instrument.index()).copied().unwrap_or(0)
    }

    /// The time of the last event any earlier window delivered.
    pub fn last_ts(&self) -> Option<i64> {
        self.last_ts
    }

    /// Windows closed so far.
    pub fn windows(&self) -> u64 {
        self.windows
    }
}

/// Why a window was not opened.
#[derive(Debug)]
pub enum WindowError {
    /// The window's first event is not strictly after the last event of the windows before it.
    Overlap { first_ts: i64, last_ts: i64 },
    /// The window is not over the run's instruments.
    Other(RaptorError),
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::Overlap { first_ts, last_ts } => write!(
                f,
                "a window starts at {first_ts}, not after {last_ts}, the last event of the one \
                 before it: a window may not cut a timestamp or overlap the one before it"
            ),
            WindowError::Other(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for WindowError {}

/// One window's events, in the order and with the numbers the whole tape's cursor gives them.
pub struct WindowCursor<'a> {
    inner: MergeCursor<'a>,
    store: &'a MarketStore,
    base: WindowBase,
    last_ts: Option<i64>,
}

impl WindowCursor<'_> {
    /// The `(timestamp, phase)` of the next event: those that share it are one group.
    #[inline]
    pub fn peek_group(&self) -> Option<(i64, u8)> {
        self.inner.peek_group()
    }

    /// The next event and its payload. `row` and `local_idx` are whole-run numbers: `row`
    /// indexes the instrument's bars, ticks or depth over every window so far.
    pub fn pull(&mut self) -> Result<Option<(EventRef, Pushed)>> {
        let Some((mut event, payload)) = RuntimeCore::pull(&mut self.inner)? else {
            return Ok(None);
        };
        let i = event.instrument;
        event.row += match event.kind {
            EventKind::Bar => self.base.bar_rows(i),
            EventKind::Trade | EventKind::Quote => self.base.tick_rows(i),
            EventKind::Depth => self.base.depth_rows(i),
        };
        event.local_idx += self.base.events(i);
        self.last_ts = Some(self.last_ts.map_or(event.ts, |t| t.max(event.ts)));
        Ok(Some((event, payload)))
    }

    /// Whether every event of the window has been pulled.
    pub fn is_drained(&self) -> bool {
        self.inner.is_drained()
    }

    /// The store behind the window: the rows (all of them, with or without events) the window
    /// holds, for a driver that keeps its own rings of them.
    pub fn store(&self) -> &MarketStore {
        self.store
    }

    /// What the windows before this one delivered.
    pub fn base(&self) -> &WindowBase {
        &self.base
    }
}

impl RuntimeCore {
    /// Open the next window of the run over `store`, which holds that window's rows only, for
    /// the same instruments in the same order as the run's. Events come off the returned cursor
    /// as the whole run's cursor would give them; instruments nobody subscribes to are skipped.
    ///
    /// Refused with [`WindowError::Overlap`] when the window starts at or before the last event
    /// of the windows before it. Nothing changes then, and the window is not counted.
    pub fn advance_window<'a>(
        &mut self,
        store: &'a MarketStore,
    ) -> std::result::Result<WindowCursor<'a>, WindowError> {
        if store.len() != self.store().len() {
            return Err(WindowError::Other(RaptorError::invalid_parameter(format!(
                "a window holds {} instruments, the run has {}",
                store.len(),
                self.store().len()
            ))));
        }
        self.window.grow_to(store.len());
        let keep: Vec<bool> = (0..store.len())
            .map(|i| !self.subscribers(InstrumentId::new(i as u32)).is_empty())
            .collect();
        let inner = MergeCursor::filtered(store, |id| keep[id.index()]);
        if let (Some((first_ts, _)), Some(last_ts)) = (inner.peek_group(), self.window.last_ts) {
            if first_ts <= last_ts {
                return Err(WindowError::Overlap { first_ts, last_ts });
            }
        }
        Ok(WindowCursor { inner, store, base: self.window.clone(), last_ts: None })
    }

    /// Close the window `cursor` was read to the end of: the next window continues the rows,
    /// the ordinals and the clock from here.
    pub fn end_window(&mut self, cursor: WindowCursor<'_>) -> Result<()> {
        if !cursor.is_drained() {
            return Err(RaptorError::invalid_parameter(
                "a window is closed once every event of it has been pulled",
            ));
        }
        let store = cursor.store;
        for i in 0..store.len() {
            let id = InstrumentId::new(i as u32);
            self.window.bars[i] += store.bars(id).map_or(0, |b| b.len() as u64);
            self.window.ticks[i] += store.ticks(id).map_or(0, |t| t.len() as u64);
            self.window.depth[i] += store.depth(id).map_or(0, |d| d.len() as u64);
            self.window.events[i] += cursor.inner.delivered(id);
        }
        if let Some(t) = cursor.last_ts {
            self.window.last_ts = Some(self.window.last_ts.map_or(t, |p| p.max(t)));
        }
        self.window.windows += 1;
        Ok(())
    }

    /// What the windows closed so far delivered.
    pub fn window_base(&self) -> &WindowBase {
        &self.window
    }
}

#[cfg(test)]
#[path = "window_tests.rs"]
mod tests;
