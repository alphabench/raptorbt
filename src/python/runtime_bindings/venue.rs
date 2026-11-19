//! A live node whose orders leave the engine.
//!
//! Without an adapter every order is matched in-process, exactly as in a
//! backtest. With one, a strategy's typed orders become [`OrderRequest`]s sent
//! to the venue, and the book hears what became of them only from
//! [`ExecutionReport`]s: those a simulated venue produces as the market moves
//! (`poll`), and those a real venue's connection hands to `push_report`.
//! Both end in `deliver`, the one place a report changes the book.
//!
//! Everything a backtest accepts is turned into orders here, so a strategy
//! that passes its backtest runs the same on a venue:
//!
//! * a size given as a capital fraction, or no size at all, is resolved to
//!   units when the order is sent, from the book as it stands on the event;
//! * `enter()` and `close_position()` become market orders (a close is
//!   reduce-only);
//! * a stop or target on an order or an entry, or configured on the position,
//!   becomes protective orders sent when the position opens: a stop, a target,
//!   linked one-cancels-other (`protective="venue"`), or levels the engine
//!   watches and answers with a market order (`protective="local"`, and for a
//!   leg the venue says it cannot hold);
//! * the session square-off sends the closing orders at the cutoff;
//! * a `Twap` sends its slices as market orders on its schedule;
//! * a halt cancels what is working to add exposure, refuses new entries,
//!   and with `flatten` sends closing orders.

use super::*;

use crate::execution::adapter::{
    client_order_id, AdapterError, ExecutionAdapter, OrderChange, OrderRequest,
};
use crate::execution::orders::{ExecutionReport, OrderSide, OrderStatus, ReportOutcome};
use crate::portfolio::session::ScheduleEntry;
use crate::portfolio::VenueError;

use super::super::strategy_bindings::parse_order;

/// An order the venue has been given.
#[derive(Clone)]
struct Placed {
    sid: usize,
    local: usize,
    order_id: u64,
    request: OrderRequest,
    /// The adapter was unavailable when the request was sent: the venue has
    /// not seen it. The same client id again sends it again.
    unsent: bool,
    /// The order ended and is queued in `Ledger::ended` for eviction.
    ended: bool,
    /// The venue made the order itself (an exit rule it holds): the engine
    /// never sent it and never resends it; only its reports are booked.
    external: bool,
}

/// How many ended orders the route keeps entries for. An id sent again, or a
/// late or repeated report, for one of the last `RETAINED_ENDED` orders to end
/// finds it; past that the entries are dropped so a long run holds a bounded
/// number of orders. A report for a dropped order is `late`; a client id used
/// again after its order was dropped is a new order.
const RETAINED_ENDED: usize = 4096;

/// The `tif` of an order the engine builds (entry, close, protective leg, slice): resolved
/// when it is sent to the instrument's [`EngineKernel::engine_tif`].
///
/// [`EngineKernel::engine_tif`]: crate::portfolio::EngineKernel::engine_tif
const TIF_ENGINE: &str = "engine";

#[derive(Default)]
struct Ledger {
    /// Venue ids of ended orders, oldest first.
    ended: std::collections::VecDeque<String>,
    /// Orders sent so far per strategy, the `seq` of the next id.
    seq: Vec<u64>,
    /// Synthesised close orders so far per strategy.
    closes: Vec<u64>,
    by_venue: HashMap<String, Placed>,
    /// `(strategy, the id the strategy gave the order)` -> venue id.
    by_client: HashMap<(usize, String), String>,
    /// `(strategy, local instrument, engine order id)` -> venue id.
    by_engine: HashMap<(usize, usize, u64), String>,
    /// Protective and close orders sent so far per strategy, for their ids.
    prot: Vec<u64>,
    /// Levels the engine watches, in place of orders at the venue.
    watchers: Vec<Watch>,
    /// One-cancels-other groups the engine keeps, for a venue with no link.
    oco: Vec<Vec<String>>,
    /// `(strategy, local instrument)` whose exit was decided while its entry
    /// was still working: the position does not exist to close yet. The exit
    /// is sent the moment the entry's fill is booked, so it meets the venue on
    /// the event the batch would stash it on.
    held_exits: std::collections::HashSet<(usize, usize)>,
    /// Exits of hedge legs the engine holds until the sold legs of their group are closed.
    covers: Vec<HeldCover>,
    /// Protective legs whose day order lapsed at a session's close with their position still
    /// open: sent again at the first event of the next session ([`VenueRoute::rearm`]).
    lapsed: Vec<Lapsed>,
    /// Re-armed legs the venue still works, by `(strategy, local instrument, engine order id)`:
    /// a later rejection of one leaves its level to the engine ([`VenueRoute::note_rearm_end`]).
    rearmed: HashMap<(usize, usize, u64), Lapsed>,
}

/// A protective leg the venue let lapse with its session (a day order on a segment that holds no
/// good-till-cancelled one), waiting to be sent again.
#[derive(Clone)]
struct Lapsed {
    sid: usize,
    local: usize,
    /// The ledger id of the position the leg protected (under netting too: a position closed and
    /// opened again is another position, with its own protection).
    position: u64,
    /// The side of the order that closes the position.
    exit: OrderSide,
    leg: LapsedLeg,
    /// The lapsed order's venue id and the engine's client id for it, for the record.
    venue_id: String,
    client_id: String,
    /// The session close the order lapsed at: it is sent again at the first event after it that
    /// the instrument's session holds.
    after_ns: i64,
    /// The venue was unavailable when the leg was sent again: the event it first failed on, and
    /// the engine's id of the order that was not sent, which is sent again under the same id.
    deferred: Option<(i64, u64)>,
}

#[derive(Clone, Copy)]
enum LapsedLeg {
    Stop(f64),
    Target(f64),
    /// A trailing stop, with the extreme it had trailed from when it lapsed.
    Trail {
        offset: crate::execution::orders::TrailOffset,
        watermark: Option<f64>,
    },
}

impl LapsedLeg {
    fn name(&self) -> &'static str {
        match self {
            LapsedLeg::Stop(_) => "stop",
            LapsedLeg::Target(_) => "target",
            LapsedLeg::Trail { .. } => "trail",
        }
    }

    fn is_stop(&self) -> bool {
        !matches!(self, LapsedLeg::Target(_))
    }

    /// The price the leg stood at: a trailing stop's at its watermark, or `None` before it has one.
    fn level(&self, exit: OrderSide) -> Option<f64> {
        match *self {
            LapsedLeg::Stop(p) | LapsedLeg::Target(p) => Some(p),
            LapsedLeg::Trail { offset, watermark } => watermark.map(|w| match exit {
                OrderSide::Sell => w - offset.at(w),
                OrderSide::Buy => w + offset.at(w),
            }),
        }
    }
}

/// Whether a protective leg that has just ended lapsed with its session rather than being taken
/// off: it expired, or (a venue with no expiry status reports a lapsed day order as cancelled)
/// it was cancelled without the engine asking, between the close of the order's session and the
/// next open. A leg the engine or the strategy asked to cancel (`PendingCancel` before the
/// report) never lapsed.
fn lapsed_by_session(
    before: OrderStatus,
    after: OrderStatus,
    report_ts: i64,
    day_close: Option<i64>,
    next_open: Option<i64>,
) -> bool {
    if before == OrderStatus::PendingCancel || before.is_terminal() {
        return false;
    }
    match after {
        OrderStatus::Expired => true,
        OrderStatus::Canceled => day_close.is_some_and(|close| {
            report_ts >= close && next_open.is_none_or(|open| report_ts <= open)
        }),
        _ => false,
    }
}

/// Whether a lapsed leg is sent again on an event at `ts`: the first event after the close it
/// lapsed at that the instrument's session holds (`in_session`).
fn rearm_due(after_ns: i64, ts: i64, in_session: bool) -> bool {
    ts > after_ns && in_session
}

/// What a journal record of a lapsed leg says about it (see [`VenueRoute::journal_rearm`]).
struct Rearm {
    reason: &'static str,
    why: Option<String>,
    qty: f64,
    /// The engine's id of the order the leg was sent as, when one was made.
    order_id: Option<u64>,
    /// The venue id of that order, when known without the engine's id.
    venue: Option<String>,
    /// The engine watches the leg's level.
    watching: bool,
    /// The bar that traded through the level: open, high, low, close.
    bar: Option<(f64, f64, f64, f64)>,
}

impl Rearm {
    fn new(reason: &'static str) -> Self {
        Rearm {
            reason,
            why: None,
            qty: 0.0,
            order_id: None,
            venue: None,
            watching: false,
            bar: None,
        }
    }
    fn why(mut self, why: &str) -> Self {
        self.why = Some(why.to_string());
        self
    }
    fn qty(mut self, qty: f64) -> Self {
        self.qty = qty;
        self
    }
    fn order_id(mut self, id: u64) -> Self {
        self.order_id = Some(id);
        self
    }
    fn venue(mut self, venue_id: &str) -> Self {
        self.venue = Some(venue_id.to_string());
        self
    }
    fn watching(mut self) -> Self {
        self.watching = true;
        self
    }
    fn bar(mut self, bar: &KernelBar) -> Self {
        self.bar = Some((bar.open, bar.high, bar.low, bar.close));
        self
    }
}

impl Lapsed {
    /// Whether `bar` traded through the leg's level: a long's stop at or below its low, its target
    /// at or above its high; the other way round for a short.
    fn crossed_by(&self, bar: &KernelBar) -> bool {
        let Some(level) = self.leg.level(self.exit) else { return false };
        leg_crossed(self.leg.is_stop(), self.exit, level, bar.high, bar.low)
    }
}

/// Whether a bar of `high` and `low` traded through a protective level (see
/// [`Lapsed::crossed_by`]). `exit` is the side of the order that closes the position.
fn leg_crossed(stop: bool, exit: OrderSide, level: f64, high: f64, low: f64) -> bool {
    let sell = exit == OrderSide::Sell;
    match (stop, sell) {
        (true, true) | (false, false) => low <= level,
        (true, false) | (false, true) => high >= level,
    }
}

/// The exit of a hedge leg waiting for the sold legs of its group to close.
#[derive(Clone)]
struct HeldCover {
    sid: usize,
    local: usize,
    position_id: u64,
    /// The sold legs left bare by closing the hedge, `(local instrument, position id)`: the
    /// hedge's close waits until none is open.
    sold: Vec<(usize, u64)>,
}

/// A protective level the engine watches (`protective="local"`).
#[derive(Clone)]
struct Watch {
    sid: usize,
    local: usize,
    /// The side of the order that closes the position.
    exit: OrderSide,
    qty: f64,
    stop: Option<f64>,
    target: Option<f64>,
    /// Under hedging, the position the level protects.
    position_id: Option<u64>,
    /// A trailing stop the engine carries: the fraction of the highest price since entry it
    /// stands below, as a backtest trails it (ratcheted after each bar's check), with the entry
    /// price and the highest price so far.
    trail: Option<Trail>,
}

#[derive(Clone, Copy)]
struct Trail {
    fraction: f64,
    entry: f64,
    high: f64,
}

/// The venue a node sends its orders to, and what it has sent.
pub(super) struct VenueRoute {
    adapter: RefCell<Box<dyn ExecutionAdapter>>,
    /// The adapter simulates an exchange that advances with the market.
    simulated: bool,
    /// A protective order resting at the venue counts against the position: an order that closes
    /// it is refused until the protection is taken off (the adapter says so).
    exits_hold_line: bool,
    account_id: Option<String>,
    /// An exception a Python adapter raised that is not a refusal, kept to
    /// be re-raised as it was.
    raised: Rc<RefCell<Option<PyErr>>>,
    ledger: RefCell<Ledger>,
    /// Protective orders are watched by the engine (`protective="local"`).
    protective_local: bool,
    /// The adapter said it cannot take the last order sent.
    unsupported_sent: std::cell::Cell<bool>,
    /// The engine's id of the last order the adapter was unavailable for (it is marked unsent).
    unavailable_sent: std::cell::Cell<Option<u64>>,
    /// The engine's id and the reason of the last order refused before or at sending.
    last_refusal: RefCell<Option<(u64, String)>>,
    /// Reports are being booked for a step that matches before any hook hears: the events they
    /// made are kept in `held` instead of reaching the strategy's hooks.
    hold: std::cell::Cell<bool>,
    /// The local instrument and the events a held report made, in the order they were booked.
    held: RefCell<HeldEvents>,
}

fn venue_err(e: VenueError) -> PyErr {
    match e {
        VenueError::FillNotBooked { .. } => {
            Fail::new("venue_fill_not_booked", e.to_string()).into()
        }
        VenueError::Unsupported(what) => Fail::new("unsupported_in_venue_run", what).into(),
        VenueError::NotReversible { .. } => {
            Fail::new("correction_not_reversible", e.to_string()).into()
        }
        other => Fail::new("invalid_execution_report", other.to_string()).into(),
    }
}

/// The local instrument and the events a booked report made.
pub(super) type HeldEvents = Vec<(usize, Vec<crate::portfolio::EngineEvent>)>;

/// What a simulated venue reported for an event: the venue's order id and the report.
pub(super) type VenueReports = Vec<(String, ExecutionReport)>;

impl VenueRoute {
    pub(super) fn new(
        mut adapter: Box<dyn ExecutionAdapter>,
        account_id: Option<String>,
        raised: Rc<RefCell<Option<PyErr>>>,
        strategies: usize,
        protective_local: bool,
    ) -> Self {
        let simulated = adapter.as_simulation().is_some();
        let exits_hold_line = !simulated && adapter.resting_exits_hold_the_line();
        Self {
            adapter: RefCell::new(adapter),
            simulated,
            exits_hold_line,
            account_id,
            raised,
            ledger: RefCell::new(Ledger {
                seq: vec![0; strategies],
                closes: vec![0; strategies],
                prot: vec![0; strategies],
                ..Ledger::default()
            }),
            protective_local,
            unsupported_sent: std::cell::Cell::new(false),
            unavailable_sent: std::cell::Cell::new(None),
            last_refusal: RefCell::new(None),
            hold: std::cell::Cell::new(false),
            held: RefCell::new(Vec::new()),
        }
    }

    /// The adapter layer stopped a request: a newer epoch owns the account, or
    /// the journal could not record it. Nothing was sent.
    /// Tell a simulated venue about an instrument added to strategy `strategy` after the node
    /// started, so its shadow book steps and takes orders on it. A venue that is not a
    /// simulation (a real broker connection) is told nothing: orders name the symbol.
    pub(super) fn add_instrument(
        &self,
        strategy: usize,
        symbol: &str,
        direction: crate::core::types::Direction,
        spec: Option<&crate::instruments::InstrumentSpec>,
        policy: crate::portfolio::ledger::PositionPolicy,
    ) -> PyResult<()> {
        let mut adapter = self.adapter.borrow_mut();
        let Some(venue) = adapter.as_simulation() else { return Ok(()) };
        venue
            .add_instrument(strategy, symbol.to_string(), direction, spec.cloned(), None, policy)
            .map_err(|e| Fail::new("invalid_execution", e.to_string()).into())
    }

    /// Whether the engine watches a protective level on this strategy's local instrument.
    pub(super) fn watches(&self, strategy: usize, local: usize) -> bool {
        self.ledger.borrow().watchers.iter().any(|w| w.sid == strategy && w.local == local)
    }

    fn stopped(&self, e: AdapterError) -> PyErr {
        match e {
            AdapterError::Fenced(why) => Fail::new("fenced", why).into(),
            AdapterError::Journal(why) => Fail::new("journal_failed", why).into(),
            other => Fail::new("venue_unavailable", other.to_string()).into(),
        }
    }

    fn unavailable(&self, message: String) -> PyErr {
        match self.raised.borrow_mut().take() {
            Some(original) => original,
            None => Fail::new("venue_unavailable", message).into(),
        }
    }

    fn placed(&self, venue_id: &str) -> Option<Placed> {
        self.ledger.borrow().by_venue.get(venue_id).cloned()
    }

    fn venue_id_of(&self, sid: StrategyId, local: usize, order_id: u64) -> Option<String> {
        self.ledger.borrow().by_engine.get(&(sid.index(), local, order_id)).cloned()
    }

    /// Whether `venue_id` is an id this route issued whose order has since
    /// been dropped from the ledger.
    pub(super) fn is_dropped(&self, venue_id: &str) -> bool {
        let Some((strategy, seq)) = crate::execution::adapter::parse_client_order_id(venue_id)
        else {
            return false;
        };
        let ledger = self.ledger.borrow();
        ledger.seq.get(strategy as usize).is_some_and(|issued| seq < *issued)
            && !ledger.by_venue.contains_key(venue_id)
    }

    /// Note that an order ended, and drop the oldest ended orders past
    /// [`RETAINED_ENDED`].
    fn note_ended(&self, venue_id: &str) {
        let mut ledger = self.ledger.borrow_mut();
        match ledger.by_venue.get_mut(venue_id) {
            Some(placed) if !placed.ended => placed.ended = true,
            _ => return,
        }
        ledger.ended.push_back(venue_id.to_string());
        while ledger.ended.len() > RETAINED_ENDED {
            let Some(old) = ledger.ended.pop_front() else { break };
            if let Some(placed) = ledger.by_venue.remove(&old) {
                let client = placed.request.tags.get("client_id").cloned().unwrap_or_default();
                ledger.by_client.remove(&(placed.sid, client));
                ledger.by_engine.remove(&(placed.sid, placed.local, placed.order_id));
            }
            for group in &mut ledger.oco {
                group.retain(|v| *v != old);
            }
            ledger.oco.retain(|g| g.len() > 1);
        }
    }

    /// [`Self::note_ended`] when the order has reached a final state. A close that ended without
    /// closing, on a venue where the protection was taken off to send it, leaves the position
    /// with only the levels the engine watches: that is journaled as `unprotected`.
    fn note_if_ended(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        order_id: u64,
    ) -> PyResult<()> {
        let ended = sh
            .with_book(sid, |b, _| {
                Ok(b.kernel(local)
                    .order(order_id)
                    .filter(|o| o.status.is_terminal())
                    .map(|o| (o.status, o.reduce_only, o.client_id.clone())))
            })
            .unwrap_or(None);
        let Some((status, reduce_only, client)) = ended else { return Ok(()) };
        if let Some(venue_id) = self.venue_id_of(sid, local, order_id) {
            self.note_ended(&venue_id);
        }
        let unfilled = !matches!(status, OrderStatus::Filled);
        if self.exits_hold_line && reduce_only && unfilled && client.starts_with("close-") {
            let watched = self
                .ledger
                .borrow()
                .watchers
                .iter()
                .any(|w| w.sid == sid.index() && w.local == local);
            sh.jlog_now(jk::ENGINE_EXIT, jsrc::ENGINE, None, || {
                JV::map([
                    ("strategy", JV::Int(sid.index() as i64)),
                    ("reason", JV::str("unprotected")),
                    ("close", JV::str(client.as_str())),
                    ("levels_watched", JV::Bool(watched)),
                    ("venue", JV::Bool(true)),
                ])
            })?;
        }
        Ok(())
    }

    /// A protective leg whose day order lapsed with its session while its position is still open
    /// (on a segment that holds no good-till-cancelled order the engine's own orders are day
    /// orders): its level is kept against the position and journaled, and the leg is sent again
    /// at the first event of the next session ([`VenueRoute::rearm`]). A backtest holds the level
    /// across sessions; without this the position would be left unprotected.
    #[allow(clippy::too_many_arguments)]
    fn note_lapse(
        &self,
        sh: &Shared,
        sid: StrategyId,
        placed: &Placed,
        venue_id: &str,
        before: OrderStatus,
        report_ts: i64,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        use crate::core::types::Direction;
        use crate::execution::orders::OrderKind;
        // (leg, side, quantity left, position, status, client id, session close)
        type Seen = (LapsedLeg, OrderSide, f64, u64, OrderStatus, String, i64);
        let seen: Option<Seen> = sh.with_book(sid, |b, _| {
            let k = b.kernel(placed.local);
            let Some(o) = k.order(placed.order_id) else { return Ok(None) };
            if !o.client_id.starts_with("prot-") || !o.reduce_only || !o.status.is_terminal() {
                return Ok(None);
            }
            let schedule = k.schedule();
            let close = schedule.and_then(|s| s.day_order_expiry_ns(o.submitted_ts));
            let next_open = close.and_then(|c| schedule.and_then(|s| s.next_open_ns(c)));
            if !lapsed_by_session(before, o.status, report_ts, close, next_open) {
                return Ok(None);
            }
            let leg = match o.kind {
                OrderKind::StopMarket { trigger } => LapsedLeg::Stop(trigger),
                OrderKind::Limit { price } => LapsedLeg::Target(price),
                OrderKind::TrailingStopMarket { offset } => {
                    LapsedLeg::Trail { offset, watermark: None }
                }
                // A market answer to a level is not a level: nothing to hold across sessions.
                _ => return Ok(None),
            };
            let closes_long = o.side == OrderSide::Sell;
            let Some(p) = k.position_snapshots().into_iter().find(|p| {
                o.position_id.is_none_or(|id| id == p.position_id)
                    && (p.direction == Direction::Long) == closes_long
            }) else {
                return Ok(None);
            };
            let left = match o.qty {
                crate::execution::orders::QtySpec::Units(q) => (q - o.filled_qty).max(0.0),
                _ => p.size,
            };
            // A venue without a session schedule says when the order lapsed by reporting it.
            let after = close.unwrap_or(report_ts);
            Ok(Some((leg, o.side, left, p.position_id, o.status, o.client_id.clone(), after)))
        })?;
        let Some((mut leg, exit, left, position, status, client, after_ns)) = seen else {
            return Ok(());
        };
        if let LapsedLeg::Trail { offset, .. } = leg {
            // The extreme the venue had trailed from: a simulated venue's own, else the one the
            // position has reached since its entry, as a backtest trails it.
            let simulated = if self.simulated {
                self.adapter.borrow_mut().as_simulation().and_then(|v| v.trail_watermark(venue_id))
            } else {
                None
            };
            let reached = sh.with_book(sid, |b, _| {
                Ok(b.kernel(placed.local).ledger.get(position).map(|m| {
                    if exit == OrderSide::Sell {
                        m.position.highest_since_entry
                    } else {
                        m.position.lowest_since_entry
                    }
                }))
            })?;
            leg = LapsedLeg::Trail { offset, watermark: simulated.or(reached) };
        }
        let level = leg.level(exit);
        sh.jlog_now(jk::ENGINE_EXIT, jsrc::ENGINE, Some(venue_id), || {
            JV::map([
                ("strategy", JV::Int(sid.index() as i64)),
                ("reason", JV::str("lapsed_day_order")),
                ("position", JV::Int(position as i64)),
                ("leg", JV::str(leg.name())),
                ("level", level.map_or(JV::Null, JV::Float)),
                ("qty", JV::Float(left)),
                ("order", JV::str(venue_id)),
                ("client_id", JV::str(client.as_str())),
                ("status", JV::str(format!("{status:?}").to_lowercase())),
                ("rearm_after_ns", JV::Int(after_ns)),
                ("idx", JV::Int(idx as i64)),
                ("event_ts", JV::Int(ts)),
                ("venue", JV::Bool(true)),
            ])
        })?;
        self.ledger.borrow_mut().lapsed.push(Lapsed {
            sid: sid.index(),
            local: placed.local,
            position,
            exit,
            leg,
            venue_id: venue_id.to_string(),
            client_id: client,
            after_ns,
            deferred: None,
        });
        Ok(())
    }

    /// Send again, at the first event of the next session, the protective legs whose day orders
    /// lapsed with the last one ([`VenueRoute::note_lapse`]), each for what the position holds
    /// now and not covered by a leg still working. Each outcome is journaled:
    ///
    /// * the position was closed since: dropped (`rearm_skipped`, `position_closed`); a working
    ///   close of it holds the legs back until its outcome is known;
    /// * on a venue that is not a simulation, the event's bar already traded through a level: the
    ///   engine watches the levels, which closes the position at market on this event
    ///   (`rearm_level_crossed`) as a level the engine watches does. A simulated venue meets the
    ///   leg with the bar instead ([`EngineKernel::match_rearmed`]);
    /// * sent (`rearmed`); refused by the host, the book or the venue (`rearm_rejected`, with the
    ///   reason): its level is watched by the engine; held by the engine because the venue cannot
    ///   hold it (`rearm_refused_watching`);
    /// * the venue unavailable: the leg stays waiting and the order that was not sent is sent
    ///   again, under the same id, on the next event (`rearm_deferred`, once); a leg still not sent
    ///   at the close of the session it was first deferred in is given up (`rearm_abandoned`), and
    ///   its unsent order is ended in the book.
    ///
    /// A leg leaves the waiting list only once it has one of these outcomes.
    ///
    /// [`EngineKernel::match_rearmed`]: crate::portfolio::EngineKernel::match_rearmed
    fn rearm(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        let mine = |l: &Lapsed| l.sid == sid.index() && l.local == local;
        if !self.ledger.borrow().lapsed.iter().any(mine) {
            return Ok(());
        }
        let in_session = sh.with_book(sid, |b, _| {
            Ok(b.kernel(local).schedule().is_none_or(|s| s.is_open(ts.saturating_sub(1))))
        })?;
        let mut positions: Vec<u64> = Vec::new();
        for l in self.ledger.borrow().lapsed.iter() {
            if mine(l) && rearm_due(l.after_ns, ts, in_session) && !positions.contains(&l.position)
            {
                positions.push(l.position);
            }
        }
        if positions.is_empty() {
            return Ok(());
        }
        let hedging = sh.with_book(sid, |b, _| Ok(b.kernel(local).is_hedging()))?;
        // The bar a real venue's leg would be sent after: what traded through a level in it is
        // already past the leg.
        let bar = if self.simulated { None } else { sh.current_bar(sid, local) };
        for position in positions {
            let held = sh.with_book(sid, |b, _| {
                let k = b.kernel(local);
                Ok(k.position_snapshots()
                    .into_iter()
                    .find(|p| p.position_id == position)
                    .map(|p| (p.size, p.entry_price)))
            })?;
            let waiting = held.is_some()
                && (self.close_working(sh, sid, local, position)?
                    || self.ledger.borrow().covers.iter().any(|c| {
                        (c.sid, c.local, c.position_id) == (sid.index(), local, position)
                    }));
            if waiting {
                continue;
            }
            let legs: Vec<Lapsed> = self
                .ledger
                .borrow()
                .lapsed
                .iter()
                .filter(|l| mine(l) && l.position == position)
                .cloned()
                .collect();
            let Some((size, entry_price)) = held else {
                self.drop_lapsed(sid, local, position, None);
                for l in &legs {
                    self.forget_unsent(sh, sid, local, l, idx, ts)?;
                    let rec = Rearm::new("rearm_skipped").why("position_closed");
                    self.journal_rearm(sh, sid, l, rec, idx, ts)?;
                }
                continue;
            };
            let named = if hedging { Some(position) } else { None };
            if let Some(bar) = bar {
                let crossed: Vec<&Lapsed> = legs.iter().filter(|l| l.crossed_by(&bar)).collect();
                if !crossed.is_empty() {
                    self.drop_lapsed(sid, local, position, None);
                    let exit = legs[0].exit;
                    let level_of = |stop: bool| {
                        legs.iter()
                            .rev()
                            .find(|l| l.leg.is_stop() == stop)
                            .and_then(|l| l.leg.level(l.exit))
                    };
                    // `watch` answers the levels on this same event with a market close.
                    self.ledger.borrow_mut().watchers.push(Watch {
                        sid: sid.index(),
                        local,
                        exit,
                        qty: size,
                        stop: level_of(true),
                        target: level_of(false),
                        position_id: named,
                        trail: None,
                    });
                    for l in &legs {
                        self.forget_unsent(sh, sid, local, l, idx, ts)?;
                        let rec = if crossed.iter().any(|c| std::ptr::eq(*c, l)) {
                            Rearm::new("rearm_level_crossed").qty(size).bar(&bar)
                        } else {
                            Rearm::new("rearm_skipped").why("level_crossed_closing")
                        };
                        self.journal_rearm(sh, sid, l, rec, idx, ts)?;
                    }
                    continue;
                }
            }
            let mut placed: Vec<(u64, f64)> = Vec::new();
            for want_stop in [true, false] {
                // The newest lapsed leg of the kind; earlier ones of it are replaced by it.
                let Some(l) = legs.iter().rev().find(|l| l.leg.is_stop() == want_stop) else {
                    continue;
                };
                // A leg first deferred in a session that has closed since is given up.
                if let Some((since, order_id)) = l.deferred {
                    let close = sh.with_book(sid, |b, _| {
                        Ok(b.kernel(local).schedule().and_then(|s| s.day_order_expiry_ns(since)))
                    })?;
                    if close.is_some_and(|close| ts > close) {
                        self.drop_lapsed(sid, local, position, Some(want_stop));
                        self.end_unsent(sh, sid, local, order_id, idx, ts, "rearm_abandoned")?;
                        let rec = Rearm::new("rearm_abandoned").why("venue_unavailable");
                        self.journal_rearm(sh, sid, l, rec.order_id(order_id), idx, ts)?;
                        continue;
                    }
                }
                let bare = match l.deferred {
                    // The order that was not sent is the leg: it is sent again as it was.
                    Some((_, order_id)) => self.unsent_qty(sid, local, order_id).unwrap_or(0.0),
                    None => {
                        let (covered, any) =
                            self.working_cover(sh, sid, local, named, want_stop)?;
                        // Where resting protection counts against the line, legs of a kind add
                        // up to the position; elsewhere each leg is resized to the whole position.
                        if self.exits_hold_line {
                            size - covered
                        } else if any {
                            0.0
                        } else {
                            size
                        }
                    }
                };
                if bare <= 1e-9 {
                    self.drop_lapsed(sid, local, position, Some(want_stop));
                    self.journal_rearm(
                        sh,
                        sid,
                        l,
                        Rearm::new("rearm_skipped").why("covered"),
                        idx,
                        ts,
                    )?;
                    continue;
                }
                self.unavailable_sent.set(None);
                *self.last_refusal.borrow_mut() = None;
                let sent = match l.deferred {
                    Some((_, order_id)) => self.send_unsent(sh, sid, local, order_id, ts),
                    None => match l.leg {
                        LapsedLeg::Stop(level) | LapsedLeg::Target(level) => self.place_leg_tagged(
                            sh, sid, local, idx, ts, l.exit, want_stop, bare, level, named, true,
                        ),
                        LapsedLeg::Trail { offset, watermark } => self.place_trail(
                            sh,
                            sid,
                            local,
                            idx,
                            ts,
                            l.exit,
                            bare,
                            offset,
                            watermark,
                            entry_price,
                            named,
                        ),
                    },
                };
                let sent = match sent {
                    Ok(sent) => sent,
                    Err(e) => match self.unavailable_sent.take() {
                        // The venue could not be reached: the leg waits for the next event, with
                        // the order that was not sent. The error is on the record, not raised:
                        // the protection is retried, and the strategy is not stopped for it.
                        Some(order_id) => {
                            let code = Python::attach(|py| {
                                e.value(py)
                                    .getattr("code")
                                    .ok()
                                    .and_then(|c| c.extract::<String>().ok())
                                    .unwrap_or_else(|| "venue_unavailable".to_string())
                            });
                            let first = l.deferred.is_none();
                            self.defer_lapsed(sid, local, position, want_stop, ts, order_id);
                            if first {
                                let rec = Rearm::new("rearm_deferred").why(&code).qty(bare);
                                self.journal_rearm(sh, sid, l, rec.order_id(order_id), idx, ts)?;
                            }
                            continue;
                        }
                        None => return Err(e),
                    },
                };
                self.drop_lapsed(sid, local, position, Some(want_stop));
                match sent {
                    Some(order_id) => {
                        let rejected = sh.with_book(sid, |b, _| {
                            Ok(b.kernel(local)
                                .order(order_id)
                                .is_some_and(|o| o.status == OrderStatus::Rejected))
                        })?;
                        if rejected {
                            let why = self
                                .last_refusal
                                .borrow()
                                .clone()
                                .filter(|(id, _)| *id == order_id)
                                .map(|(_, why)| why)
                                .unwrap_or_default();
                            self.watch_leg(sid, local, l, bare, named, entry_price);
                            let rec = Rearm::new("rearm_rejected").why(&why).qty(bare).watching();
                            self.journal_rearm(sh, sid, l, rec.order_id(order_id), idx, ts)?;
                        } else {
                            self.ledger
                                .borrow_mut()
                                .rearmed
                                .insert((sid.index(), local, order_id), l.clone());
                            let rec = Rearm::new("rearmed").qty(bare).order_id(order_id);
                            self.journal_rearm(sh, sid, l, rec, idx, ts)?;
                            placed.push((order_id, bare));
                        }
                    }
                    None => {
                        let rec = Rearm::new("rearm_refused_watching").qty(bare);
                        self.journal_rearm(sh, sid, l, rec, idx, ts)?
                    }
                }
            }
            // A stop and a target sent again for the same quantity cancel each other, as they did.
            if let [(a, qa), (b, qb)] = placed[..] {
                if (qa - qb).abs() < 1e-9 {
                    self.link_oco(sid, local, &[a, b])?;
                }
            }
        }
        self.resize_protection(sh, sid, local, idx)
    }

    /// Take the lapsed legs of a position off the waiting list: all, or those of one kind (`stop`).
    fn drop_lapsed(&self, sid: StrategyId, local: usize, position: u64, stop: Option<bool>) {
        self.ledger.borrow_mut().lapsed.retain(|l| {
            !(l.sid == sid.index()
                && l.local == local
                && l.position == position
                && stop.is_none_or(|s| l.leg.is_stop() == s))
        });
    }

    /// Note on the waiting legs of a kind that the venue was unavailable for `order_id`, the order
    /// that carries them now, keeping the event it was first deferred on.
    fn defer_lapsed(
        &self,
        sid: StrategyId,
        local: usize,
        position: u64,
        stop: bool,
        ts: i64,
        order_id: u64,
    ) {
        for l in self.ledger.borrow_mut().lapsed.iter_mut() {
            if l.sid == sid.index()
                && l.local == local
                && l.position == position
                && l.leg.is_stop() == stop
            {
                let since = l.deferred.map_or(ts, |(since, _)| since);
                l.deferred = Some((since, order_id));
            }
        }
    }

    /// The quantity of an order that was never sent (the venue was unavailable), while it is.
    fn unsent_qty(&self, sid: StrategyId, local: usize, order_id: u64) -> Option<f64> {
        let ledger = self.ledger.borrow();
        let venue_id = ledger.by_engine.get(&(sid.index(), local, order_id))?;
        ledger.by_venue.get(venue_id).filter(|p| p.unsent).map(|p| p.request.quantity)
    }

    /// Send again, under its own id, an order the venue was unavailable for. `Some(order_id)` once
    /// the venue has it (or it was sent since), `None` never: a leg the venue cannot hold was
    /// already watched when it was first sent.
    fn send_unsent(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        order_id: u64,
        ts: i64,
    ) -> PyResult<Option<u64>> {
        let placed = {
            let ledger = self.ledger.borrow();
            ledger
                .by_engine
                .get(&(sid.index(), local, order_id))
                .and_then(|v| ledger.by_venue.get(v))
                .cloned()
        };
        if let Some(p) = placed.filter(|p| p.unsent) {
            self.send(sh, sid, local, order_id, ts, p.request.submitted_idx, &p.request)?;
        }
        Ok(Some(order_id))
    }

    /// End in the book an order the venue was never sent (unavailable), so it is not counted as
    /// protection: it is rejected with `reason`, as the venue never took it.
    #[allow(clippy::too_many_arguments)]
    fn end_unsent(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        order_id: u64,
        idx: usize,
        ts: i64,
        reason: &str,
    ) -> PyResult<()> {
        if self.unsent_qty(sid, local, order_id).is_none() {
            return Ok(());
        }
        let working = sh.with_book(sid, |b, _| {
            Ok(b.kernel(local).order(order_id).is_some_and(|o| !o.status.is_terminal()))
        })?;
        if !working {
            return Ok(());
        }
        self.book_silently(
            sh,
            sid,
            local,
            order_id,
            idx,
            ts,
            &ExecutionReport::Rejected {
                reason: format!("{reason}: the venue was unavailable"),
                ts_ns: ts,
            },
        )
    }

    /// End the unsent order a deferred leg carries, when it has one.
    fn forget_unsent(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        l: &Lapsed,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        match l.deferred {
            Some((_, order_id)) => {
                self.end_unsent(sh, sid, local, order_id, idx, ts, "rearm_dropped")
            }
            None => Ok(()),
        }
    }

    /// Leave a lapsed leg's level to the engine, which watches it for `qty` (a trailing stop from
    /// the extreme it had reached) and closes at market when it is reached.
    fn watch_leg(
        &self,
        sid: StrategyId,
        local: usize,
        l: &Lapsed,
        qty: f64,
        named: Option<u64>,
        entry_price: f64,
    ) {
        use crate::execution::orders::TrailOffset;
        let mut watch = Watch {
            sid: sid.index(),
            local,
            exit: l.exit,
            qty,
            stop: None,
            target: None,
            position_id: named,
            trail: None,
        };
        match l.leg {
            LapsedLeg::Stop(level) => watch.stop = Some(level),
            LapsedLeg::Target(level) => watch.target = Some(level),
            LapsedLeg::Trail { offset, watermark } => {
                let high = watermark.unwrap_or(entry_price);
                let fraction = match offset {
                    TrailOffset::Bps(bps) => bps / 10_000.0,
                    TrailOffset::Price(price) if entry_price > 0.0 => price / entry_price,
                    TrailOffset::Price(_) => 0.0,
                };
                watch.stop = LapsedLeg::Trail { offset, watermark: Some(high) }.level(l.exit);
                watch.trail =
                    (fraction > 0.0).then_some(Trail { fraction, entry: entry_price, high });
            }
        }
        self.ledger.borrow_mut().watchers.push(watch);
    }

    /// A re-armed leg ended. A rejection the venue reported after taking it (a broker refuses a
    /// stop whose trigger the market has already passed) leaves its level to the engine, which
    /// watches it and, the level being passed, closes at market on the next event; journaled
    /// `rearm_rejected` with the venue's reason.
    #[allow(clippy::too_many_arguments)]
    fn note_rearm_end(
        &self,
        sh: &Shared,
        sid: StrategyId,
        placed: &Placed,
        venue_id: &str,
        before: OrderStatus,
        report: &ExecutionReport,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        let key = (sid.index(), placed.local, placed.order_id);
        if !self.ledger.borrow().rearmed.contains_key(&key) {
            return Ok(());
        }
        let ended = sh.with_book(sid, |b, _| {
            Ok(b.kernel(placed.local).order(placed.order_id).map(|o| {
                let left = match o.qty {
                    crate::execution::orders::QtySpec::Units(q) => (q - o.filled_qty).max(0.0),
                    _ => 0.0,
                };
                (o.status, left)
            }))
        })?;
        let Some((status, left)) = ended else { return Ok(()) };
        if !status.is_terminal() {
            return Ok(());
        }
        let Some(l) = self.ledger.borrow_mut().rearmed.remove(&key) else { return Ok(()) };
        let ExecutionReport::Rejected { reason, .. } = report else { return Ok(()) };
        if status != OrderStatus::Rejected || before.is_terminal() {
            return Ok(());
        }
        let held = sh.with_book(sid, |b, _| {
            let k = b.kernel(placed.local);
            Ok(k.position_snapshots()
                .into_iter()
                .find(|p| p.position_id == l.position)
                .map(|p| (p.size, p.entry_price, k.is_hedging())))
        })?;
        let Some((size, entry_price, hedging)) = held else { return Ok(()) };
        let named = if hedging { Some(l.position) } else { None };
        let qty = if self.exits_hold_line { left.min(size) } else { size };
        self.watch_leg(sid, placed.local, &l, qty, named, entry_price);
        let rec = Rearm::new("rearm_rejected").why(reason).qty(qty).watching();
        self.journal_rearm(sh, sid, &l, rec.venue(venue_id), idx, ts)
    }

    /// The quantity the working protective legs of a kind (stops, trailing ones included, or
    /// targets) still have to close for a position (`named` under hedging), and whether any works.
    fn working_cover(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        named: Option<u64>,
        stop: bool,
    ) -> PyResult<(f64, bool)> {
        use crate::execution::orders::OrderKind;
        sh.with_book(sid, |b, _| {
            let mut left = 0.0;
            let mut any = false;
            for o in b.kernel(local).open_orders() {
                let kind_stop = match o.kind {
                    OrderKind::StopMarket { .. } | OrderKind::TrailingStopMarket { .. } => true,
                    OrderKind::Limit { .. } => false,
                    _ => continue,
                };
                if !o.client_id.starts_with("prot-")
                    || !o.reduce_only
                    || kind_stop != stop
                    || o.status == OrderStatus::PendingCancel
                    || named.is_some_and(|p| o.position_id.is_some_and(|q| q != p))
                {
                    continue;
                }
                any = true;
                if let crate::execution::orders::QtySpec::Units(q) = o.qty {
                    left += (q - o.filled_qty).max(0.0);
                }
            }
            Ok((left, any))
        })
    }

    /// Send a lapsed trailing stop again: a trailing order of the same offset, which a simulated
    /// venue starts from the extreme the lapsed one had reached. A venue that cannot hold it
    /// leaves the level to the engine, which trails it from that extreme. The id of the order
    /// when the venue holds it.
    #[allow(clippy::too_many_arguments)]
    fn place_trail(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        ts: i64,
        exit: OrderSide,
        qty: f64,
        offset: crate::execution::orders::TrailOffset,
        watermark: Option<f64>,
        entry_price: f64,
        position_id: Option<u64>,
    ) -> PyResult<Option<u64>> {
        use crate::execution::orders::TrailOffset;
        let mut spec = Self::spec(exit, "trailing_stop_market", Some(qty), true);
        match offset {
            TrailOffset::Bps(bps) => {
                spec.offset = Some(bps);
                spec.offset_kind = "bps".to_string();
            }
            TrailOffset::Price(price) => spec.offset = Some(price),
        }
        spec.position_id = position_id;
        spec.tags.insert(crate::execution::adapter::REARM_TAG.to_string(), "1".to_string());
        if let Some(w) = watermark {
            spec.tags.insert(crate::execution::adapter::TRAIL_FROM_TAG.to_string(), w.to_string());
        }
        let n = self.next_prot(sid);
        self.unsupported_sent.set(false);
        let id = self.submit(sh, sid, local, &spec, idx, ts, &format!("prot-{n}"), None)?;
        if !self.unsupported_sent.replace(false) {
            return Ok(Some(id));
        }
        let high = watermark.unwrap_or(entry_price);
        let fraction = match offset {
            TrailOffset::Bps(bps) => bps / 10_000.0,
            TrailOffset::Price(price) if entry_price > 0.0 => price / entry_price,
            TrailOffset::Price(_) => 0.0,
        };
        let level = match exit {
            OrderSide::Sell => high - offset.at(high),
            OrderSide::Buy => high + offset.at(high),
        };
        self.ledger.borrow_mut().watchers.push(Watch {
            sid: sid.index(),
            local,
            exit,
            qty,
            stop: Some(level),
            target: None,
            position_id,
            trail: (fraction > 0.0).then_some(Trail { fraction, entry: entry_price, high }),
        });
        Ok(None)
    }

    /// The journal record of what became of a lapsed leg (see [`VenueRoute::rearm`]): the leg it
    /// replaces, its level, and the new order (its venue id when it was entered with the venue, and
    /// the engine's client id even when it was refused before that).
    fn journal_rearm(
        &self,
        sh: &Shared,
        sid: StrategyId,
        l: &Lapsed,
        rec: Rearm,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        let venue_id = rec
            .venue
            .clone()
            .or_else(|| rec.order_id.and_then(|id| self.venue_id_of(sid, l.local, id)));
        let client = match rec.order_id {
            Some(id) => sh.with_book(sid, |b, _| {
                Ok(b.kernel(l.local).order(id).map(|o| o.client_id.clone()))
            })?,
            None => venue_id
                .as_deref()
                .and_then(|v| self.placed(v))
                .and_then(|p| p.request.tags.get("client_id").cloned()),
        };
        sh.jlog_now(jk::ENGINE_EXIT, jsrc::ENGINE, venue_id.as_deref(), || {
            let mut v = JV::map([
                ("strategy", JV::Int(sid.index() as i64)),
                ("reason", JV::str(rec.reason)),
                ("why", rec.why.map_or(JV::Null, JV::str)),
                ("position", JV::Int(l.position as i64)),
                ("leg", JV::str(l.leg.name())),
                ("level", l.leg.level(l.exit).map_or(JV::Null, JV::Float)),
                ("qty", JV::Float(rec.qty)),
                ("order", venue_id.clone().map_or(JV::Null, JV::str)),
                ("client_id", client.map_or(JV::Null, JV::str)),
                ("replaces", JV::str(l.venue_id.as_str())),
                ("replaces_client_id", JV::str(l.client_id.as_str())),
                ("idx", JV::Int(idx as i64)),
                ("event_ts", JV::Int(ts)),
                ("venue", JV::Bool(true)),
            ]);
            if let JV::Map(m) = &mut v {
                if rec.watching {
                    m.insert("watching".to_string(), JV::Bool(true));
                }
                if let Some((o, h, lo, c)) = rec.bar {
                    m.insert(
                        "bar".to_string(),
                        JV::map([
                            ("open", JV::Float(o)),
                            ("high", JV::Float(h)),
                            ("low", JV::Float(lo)),
                            ("close", JV::Float(c)),
                        ]),
                    );
                }
            }
            v
        })
    }

    /// The protective legs waiting to be sent again after their day order lapsed:
    /// `(strategy, local instrument, position id, leg, level)`, for the node's state digest.
    pub(super) fn lapsed_legs(&self) -> Vec<(usize, usize, u64, String, f64)> {
        self.ledger
            .borrow()
            .lapsed
            .iter()
            .map(|l| {
                (
                    l.sid,
                    l.local,
                    l.position,
                    l.leg.name().to_string(),
                    l.leg.level(l.exit).unwrap_or(0.0),
                )
            })
            .collect()
    }

    // -- requests --------------------------------------------------------

    /// The quantity of an order that names no units: its parent's, for a
    /// held leg; what the book would size it to from this event.
    fn resolve_units(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        parent: Option<u64>,
        spec: &OrderSpec,
        tick: f64,
    ) -> PyResult<Result<f64, String>> {
        if spec.size_frac.is_none() {
            let ledger = self.ledger.borrow();
            if let Some(p) = parent {
                if let Some(known) = ledger
                    .by_engine
                    .get(&(sid.index(), local, p))
                    .and_then(|v| ledger.by_venue.get(v))
                {
                    return Ok(Ok(known.request.quantity));
                }
            }
        }
        let parsed = parse_order(
            tick,
            &spec.side,
            &spec.kind,
            Some(1.0),
            None,
            spec.limit_price,
            spec.trigger_price,
            &spec.tif,
            spec.expire_ns,
            spec.offset,
            &spec.offset_kind,
            spec.limit_offset,
            spec.post_only,
        )?;
        let Some(bar) = sh.current_bar(sid, local) else {
            return Ok(Err("zero_size".to_string()));
        };
        // Where the order is expected to trade: its limit or trigger, else
        // the fill-price model on the event.
        let reference = parsed.kind.limit_price().or(parsed.kind.trigger_price());
        let limit_priced = matches!(parsed.kind, crate::execution::orders::OrderKind::Limit { .. });
        let units = sh.with_book_mut(sid, |b, _| {
            Ok(b.venue_units(local, &bar, parsed.side, spec.size_frac, reference, limit_priced))
        })?;
        Ok(units.map_err(str::to_string))
    }

    /// Send a new order. Returns the engine's id for it. The same
    /// `client_id` again, with the same terms, is the same order: nothing is
    /// sent twice.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn submit(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        spec: &OrderSpec,
        idx: usize,
        ts: i64,
        client_id: &str,
        parent: Option<u64>,
    ) -> PyResult<u64> {
        if sh.warmup.get() && !self.simulated {
            return Err(Fail::new(
                "unsupported_in_venue_run",
                "an order was made while replaying warmup bars, and a real venue is never sent \
                 an order made on history: do not trade on warmup bars (warmup_bars= is for \
                 priming indicators), or run on PaperExecution",
            )
            .param("warmup")
            .into());
        }
        let global = sh.global_of(sid, local)?;
        let symbol = sh.symbol_text(global.index());
        let tick = sh.with_book(sid, |b, _| Ok(b.kernel(local).price_increment()))?;
        // An order that names no time in force takes its instrument's default; one the engine
        // built takes gtc only where the instrument's segment holds it.
        let resolved;
        let spec =
            if spec.tif == super::super::strategy_bindings::TIF_AUTO || spec.tif == TIF_ENGINE {
                let engine = spec.tif == TIF_ENGINE;
                let tif = sh.with_book(sid, |b, _| {
                    let k = b.kernel(local);
                    Ok(if engine { k.engine_tif() } else { k.default_tif() })
                })?;
                resolved = OrderSpec { tif: tif.to_string(), ..spec.clone() };
                &resolved
            } else {
                spec
            };
        if let Err(why) =
            sh.with_book(sid, |b, _| Ok(b.kernel(local).check_validity(&spec.tif, spec.expire_ns)))?
        {
            return Err(Fail::new("order_validity_unsupported", why)
                .param("tif")
                .value(spec.tif.as_str())
                .valid(["day", "ioc", "fok"])
                .into());
        }

        // The same id again with the same terms is the same order: it is not
        // sent twice. A repeated order with no size keeps the size it had.
        let known = {
            let ledger = self.ledger.borrow();
            ledger
                .by_client
                .get(&(sid.index(), client_id.to_string()))
                .and_then(|v| ledger.by_venue.get(v))
                .cloned()
        };
        if let Some(known) = known {
            let units = spec.units.unwrap_or(known.request.quantity);
            let parsed = parse_order(
                tick,
                &spec.side,
                &spec.kind,
                Some(units),
                None,
                spec.limit_price,
                spec.trigger_price,
                &spec.tif,
                spec.expire_ns,
                spec.offset,
                &spec.offset_kind,
                spec.limit_offset,
                spec.post_only,
            )?;
            let r = &known.request;
            return if r.symbol == symbol
                && r.side == parsed.side
                && r.quantity == units
                && r.kind == parsed.kind
                && r.tif == parsed.tif
            {
                if known.unsent {
                    // The attempt never reached the venue: this call is the
                    // retry, and the order is still the same order.
                    self.send(
                        sh,
                        sid,
                        known.local,
                        known.order_id,
                        ts,
                        known.request.submitted_idx,
                        &known.request,
                    )?;
                }
                Ok(known.order_id)
            } else {
                Err(Fail::new(
                    "duplicate_client_order_id",
                    format!("order id {client_id:?} was already used for a different order"),
                )
                .param("client_id")
                .value(client_id)
                .into())
            };
        }

        // The terms go to the venue as written, so they must already sit on
        // the instrument's grid: the simulation refuses the same orders.
        let grid = sh.with_book(sid, |b, _| Ok(b.kernel(local).order_grid()))?;
        crate::python::strategy_bindings::check_grid(
            &grid,
            spec.units,
            &[
                ("limit_price", spec.limit_price),
                ("trigger_price", spec.trigger_price),
                ("stop_price", spec.stop_price),
                ("target_price", spec.target_price),
            ],
        )?;

        // An order that names no units is sized now, from the book on this
        // event: a fraction of the capital on hand, or the whole position.
        let (units, sized_to) = match spec.units {
            Some(units) => (units, Ok(units)),
            None => match self.resolve_units(sh, sid, local, parent, spec, tick)? {
                Ok(units) => (units, Ok(units)),
                // Nothing to trade: the order is refused as the matcher would
                // refuse it, with a smallest nominal size on the record.
                Err(reason) => (f64::MIN_POSITIVE, Err(reason)),
            },
        };
        let parsed = parse_order(
            tick,
            &spec.side,
            &spec.kind,
            Some(units),
            None,
            spec.limit_price,
            spec.trigger_price,
            &spec.tif,
            spec.expire_ns,
            spec.offset,
            &spec.offset_kind,
            spec.limit_offset,
            spec.post_only,
        )?;

        let (venue_id, parent_vid) = {
            let mut ledger = self.ledger.borrow_mut();
            let parent_vid = match parent {
                None => None,
                Some(p) => Some(
                    ledger.by_engine.get(&(sid.index(), local, p)).cloned().ok_or_else(|| {
                        PyErr::from(Fail::new(
                            "invalid_order_link",
                            "the parent order is not one the venue works",
                        ))
                    })?,
                ),
            };
            let seq = ledger.seq[sid.index()];
            ledger.seq[sid.index()] += 1;
            (client_order_id(sid.index() as u32, seq), parent_vid)
        };

        // The host's own check comes before the book's, so its refusal is the
        // one the strategy hears whatever else would have refused the order. It
        // runs before the order exists: a check that raises leaves no order
        // behind that the venue never heard of.
        let refusal = sh.refusal_for(sid, local, spec, client_id, ts)?;
        let (order_id, sub_idx) = sh.with_book_mut(sid, |book, current| {
            let sub_idx = book.submission_idx(local, idx, current);
            let id = book.kernel_mut(local).submit_venue_order(
                parsed.side,
                units,
                parsed.kind,
                parsed.tif,
                sub_idx,
                ts,
                client_id.to_string(),
                spec.post_only,
                spec.reduce_only,
                parent,
                spec.stop_price,
                spec.target_price,
            );
            if spec.position_id.is_some() {
                book.kernel_mut(local).set_order_position(id, spec.position_id);
            }
            Ok((id, sub_idx))
        })?;
        if let Some(code) = refusal {
            self.book_silently(
                sh,
                sid,
                local,
                order_id,
                sub_idx,
                ts,
                &ExecutionReport::Rejected { reason: code.to_string(), ts_ns: ts },
            )?;
            return Ok(order_id);
        }
        if let Err(reason) = sized_to {
            self.book_silently(
                sh,
                sid,
                local,
                order_id,
                sub_idx,
                ts,
                &ExecutionReport::Rejected { reason, ts_ns: ts },
            )?;
            return Ok(order_id);
        }
        // A venue that is not a simulation does not know this book's limits,
        // so they are applied here, before anything is sent.
        if !self.simulated {
            let price = parsed.kind.limit_price().or(parsed.kind.trigger_price());
            // An order that names no price is priced at the last one seen.
            let mark = sh.current_bar(sid, local).map(|bar| bar.close).filter(|c| *c > 0.0);
            let verdict = sh.with_book_mut(sid, |book, _| {
                Ok(book.venue_pretrade(
                    local,
                    Some(order_id),
                    parsed.side,
                    units,
                    spec.reduce_only,
                    price,
                    mark,
                ))
            })?;
            if let Err(reason) = verdict {
                self.book_silently(
                    sh,
                    sid,
                    local,
                    order_id,
                    sub_idx,
                    ts,
                    &ExecutionReport::Rejected { reason: reason.as_str().to_string(), ts_ns: ts },
                )?;
                return Ok(order_id);
            }
        }
        // Where a resting protective order counts against the position, an order that reduces it
        // would be refused: the protection is taken off first (its cancel is sent ahead of this
        // order) and its levels are watched by the engine for what remains, and for the whole
        // position should this order not close it.
        if self.exits_hold_line
            && spec.reduce_only
            && matches!(parsed.kind, crate::execution::orders::OrderKind::Market)
        {
            let held = sh.with_book(sid, |b, _| {
                let k = b.kernel(local);
                Ok(k.position_snapshots()
                    .into_iter()
                    .find(|p| spec.position_id.is_none_or(|id| id == p.position_id))
                    .map(|p| (p.position_id, p.size, k.is_hedging())))
            })?;
            if let Some((position_id, size, hedging)) = held {
                let left = if units < size { size - units } else { size };
                let named = if hedging { Some(position_id) } else { None };
                self.take_levels(sh, sid, local, position_id, parsed.side, left, named, sub_idx)?;
            }
        }
        let mut tags = spec.tags.clone();
        // The follow-up mark tells the simulated venue which bar an answer to a fill meets; a
        // real broker decides its own timing and is not sent the engine's internal tag.
        if !self.simulated {
            tags.remove(super::FOLLOW_UP_TAG);
            tags.remove(crate::execution::adapter::REARM_TAG);
            tags.remove(crate::execution::adapter::TRAIL_FROM_TAG);
        }
        tags.insert("client_id".to_string(), client_id.to_string());

        let request = OrderRequest {
            client_order_id: venue_id.clone(),
            symbol,
            side: parsed.side,
            quantity: units,
            kind: parsed.kind,
            tif: parsed.tif,
            strategy_id: sid.index() as u32,
            account_id: self.account_id.clone(),
            reduce_only: spec.reduce_only,
            post_only: spec.post_only,
            parent_client_order_id: parent_vid,
            tags,
            submitted_ns: ts,
            submitted_idx: sub_idx,
            size_frac: spec.size_frac,
            position_id: spec.position_id,
        };
        {
            let mut ledger = self.ledger.borrow_mut();
            let placed = Placed {
                sid: sid.index(),
                local,
                order_id,
                request: request.clone(),
                unsent: false,
                ended: false,
                external: false,
            };
            ledger.by_client.insert((sid.index(), client_id.to_string()), venue_id.clone());
            ledger.by_engine.insert((sid.index(), local, order_id), venue_id.clone());
            ledger.by_venue.insert(venue_id.clone(), placed);
        }
        self.send(sh, sid, local, order_id, ts, sub_idx, &request)?;
        Ok(order_id)
    }

    /// Hand a request to the adapter. A refusal rejects the order; an
    /// adapter that is unavailable leaves the order marked unsent and raises,
    /// so sending the same client id again retries it instead of leaving an
    /// order the venue never saw `Submitted` forever.
    #[allow(clippy::too_many_arguments)]
    fn send(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        order_id: u64,
        ts: i64,
        sub_idx: usize,
        request: &OrderRequest,
    ) -> PyResult<()> {
        let sent = {
            let mut adapter = self.adapter.borrow_mut();
            if let Some(venue) = adapter.as_simulation() {
                // what the handlers of this event decided meets the bar of the instant (see
                // `PaperVenue::submit`)
                venue.set_deciding(sh.deciding.get());
            }
            adapter.submit(request)
        };
        let unsent = matches!(sent, Err(AdapterError::Unavailable(_)));
        if let Some(placed) = self.ledger.borrow_mut().by_venue.get_mut(&request.client_order_id) {
            placed.unsent = unsent;
        }
        match sent {
            Ok(()) => Ok(()),
            Err(AdapterError::Refused(why)) => {
                self.refuse(sh, sid, local, order_id, ts, sub_idx, why)
            }
            Err(AdapterError::Unsupported(why)) => {
                self.unsupported_sent.set(true);
                self.refuse(sh, sid, local, order_id, ts, sub_idx, format!("unsupported: {why}"))
            }
            Err(AdapterError::Unavailable(why)) => {
                self.unavailable_sent.set(Some(order_id));
                Err(self.unavailable(why))
            }
            Err(e @ (AdapterError::Fenced(_) | AdapterError::Journal(_))) => Err(self.stopped(e)),
        }
    }

    /// The venue will not take an order: reject it here, as the venue would
    /// have. The strategy hears of it at its next event.
    #[allow(clippy::too_many_arguments)]
    fn refuse(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        order_id: u64,
        ts: i64,
        idx: usize,
        reason: String,
    ) -> PyResult<()> {
        self.book_silently(
            sh,
            sid,
            local,
            order_id,
            idx,
            ts,
            &ExecutionReport::Rejected { reason, ts_ns: ts },
        )
    }

    /// Fold a report in and leave the events for the book's next step.
    #[allow(clippy::too_many_arguments)]
    fn book_silently(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        order_id: u64,
        idx: usize,
        ts: i64,
        report: &ExecutionReport,
    ) -> PyResult<()> {
        if let ExecutionReport::Rejected { reason, .. } = report {
            *self.last_refusal.borrow_mut() = Some((order_id, reason.clone()));
        }
        sh.with_book_mut(sid, |book, _| {
            let (_, events) =
                book.apply_venue_report(local, idx, ts, order_id, report).map_err(venue_err)?;
            book.kernel_mut(local).pending_events.extend(events);
            Ok(())
        })?;
        self.note_if_ended(sh, sid, local, order_id)
    }

    /// Ask the venue to cancel an order it works.
    pub(super) fn cancel(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        order_id: u64,
    ) -> PyResult<()> {
        let Some(venue_id) = self.venue_id_of(sid, local, order_id) else { return Ok(()) };
        let asked = sh.with_book_mut(sid, |b, _| Ok(b.request_venue_cancel(local, order_id)));
        match asked? {
            Ok(()) => {}
            // Asking twice, or for an order that is already over, is the
            // same no-op the in-process path makes.
            Err(VenueError::Order(
                crate::execution::orders::OrderEventError::AlreadyPending { .. }
                | crate::execution::orders::OrderEventError::IllegalTransition { .. },
            )) => return Ok(()),
            Err(e) => return Err(venue_err(e)),
        }
        let ts = self.placed(&venue_id).map_or(0, |p| p.request.submitted_ns);
        let sent = self.adapter.borrow_mut().cancel(&venue_id);
        match sent {
            Ok(()) => Ok(()),
            Err(AdapterError::Refused(reason) | AdapterError::Unsupported(reason)) => self
                .book_silently(
                    sh,
                    sid,
                    local,
                    order_id,
                    idx,
                    ts,
                    &ExecutionReport::CancelRejected { reason, ts_ns: ts },
                ),
            Err(AdapterError::Unavailable(why)) => Err(self.unavailable(why)),
            Err(e @ (AdapterError::Fenced(_) | AdapterError::Journal(_))) => Err(self.stopped(e)),
        }
    }

    /// Ask the venue to cancel every order of the instrument it works.
    pub(super) fn cancel_all(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
    ) -> PyResult<()> {
        let ids: Vec<u64> = sh.with_book(sid, |b, _| {
            Ok(b.kernel(local)
                .open_orders()
                .into_iter()
                .filter(|o| o.is_venue_driven() && o.status != OrderStatus::PendingCancel)
                .map(|o| o.id)
                .collect())
        })?;
        for id in ids {
            self.cancel(sh, sid, local, idx, id)?;
        }
        Ok(())
    }

    /// Ask the venue to change an order it works.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn modify(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        order_id: u64,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) -> PyResult<()> {
        if size_frac.is_some() {
            return Err(Fail::new(
                "venue_requires_units",
                "a change to an order sent to an execution adapter must give its size as units",
            )
            .param("units")
            .into());
        }
        let Some(venue_id) = self.venue_id_of(sid, local, order_id) else { return Ok(()) };
        sh.with_book_mut(sid, |b, _| Ok(b.request_venue_replace(local, order_id)))?
            .map_err(venue_err)?;
        let ts = self.placed(&venue_id).map_or(0, |p| p.request.submitted_ns);
        let change = OrderChange {
            quantity: units,
            limit_price,
            trigger_price,
            decided_close: sh.decided_close(sid, local),
        };
        let sent = self.adapter.borrow_mut().modify(&venue_id, &change);
        match sent {
            Ok(()) => Ok(()),
            Err(AdapterError::Refused(reason) | AdapterError::Unsupported(reason)) => self
                .book_silently(
                    sh,
                    sid,
                    local,
                    order_id,
                    idx,
                    ts,
                    &ExecutionReport::ReplaceRejected { reason, ts_ns: ts },
                ),
            Err(AdapterError::Unavailable(why)) => Err(self.unavailable(why)),
            Err(e @ (AdapterError::Fenced(_) | AdapterError::Journal(_))) => Err(self.stopped(e)),
        }
    }

    /// Make orders one-cancels-other: at the venue when it can, else the
    /// engine keeps the link and cancels the rest when one fills.
    pub(super) fn link_oco(
        &self,
        sid: StrategyId,
        local: usize,
        order_ids: &[u64],
    ) -> PyResult<()> {
        let ids: Vec<String> =
            order_ids.iter().filter_map(|id| self.venue_id_of(sid, local, *id)).collect();
        if ids.len() < 2 {
            return Ok(());
        }
        let linked = self.adapter.borrow_mut().link_oco(&ids);
        match linked {
            Ok(()) => Ok(()),
            Err(AdapterError::Unsupported(_)) => {
                self.ledger.borrow_mut().oco.push(ids);
                Ok(())
            }
            Err(AdapterError::Refused(why)) => Err(Fail::new("invalid_order_link", why).into()),
            Err(AdapterError::Unavailable(why)) => Err(self.unavailable(why)),
            Err(e @ (AdapterError::Fenced(_) | AdapterError::Journal(_))) => Err(self.stopped(e)),
        }
    }

    /// An order of an engine-kept one-cancels-other group filled: cancel the
    /// rest.
    fn emulate_oco(
        &self,
        sh: &Shared,
        sid: StrategyId,
        venue_id: &str,
        idx: usize,
    ) -> PyResult<()> {
        let peers: Vec<Placed> = {
            let ledger = self.ledger.borrow();
            ledger
                .oco
                .iter()
                .filter(|g| g.iter().any(|v| v == venue_id))
                .flat_map(|g| g.iter().filter(|v| *v != venue_id))
                .filter_map(|v| ledger.by_venue.get(v).cloned())
                .collect()
        };
        for peer in peers {
            self.cancel(sh, sid, peer.local, idx, peer.order_id)?;
        }
        Ok(())
    }

    /// Close a position the way a venue can: a market order against it, for
    /// its whole size, that may only reduce.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn close(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        ts: i64,
        position_id: u64,
        follow_up: bool,
    ) -> PyResult<()> {
        let held = sh.with_book(sid, |b, _| {
            Ok(b.kernel(local)
                .position_snapshots()
                .into_iter()
                .find(|p| p.position_id == position_id)
                .map(|p| (p.size, p.direction)))
        })?;
        let Some((size, direction)) = held else { return Ok(()) };
        // A hedging account closes a position by naming it: the order carries the position's id,
        // and a venue that keeps positions apart closes exactly that one. (A venue that nets
        // refuses a hedging node when it is built.)
        let hedging = sh.with_book(sid, |b, _| Ok(b.kernel(local).is_hedging()))?;
        let n = {
            let mut ledger = self.ledger.borrow_mut();
            let n = ledger.closes[sid.index()];
            ledger.closes[sid.index()] += 1;
            n
        };
        let spec = OrderSpec {
            side: if direction == crate::core::types::Direction::Long { "sell" } else { "buy" }
                .to_string(),
            kind: "market".to_string(),
            units: Some(size),
            size_frac: None,
            limit_price: None,
            trigger_price: None,
            tif: TIF_ENGINE.to_string(),
            expire_ns: None,
            stop_price: None,
            target_price: None,
            offset: None,
            offset_kind: "price".to_string(),
            limit_offset: 0.0,
            post_only: false,
            reduce_only: true,
            position_id: if hedging { Some(position_id) } else { None },
            tags: Default::default(),
        };
        let mut spec = spec;
        if follow_up {
            spec.tags.insert(super::FOLLOW_UP_TAG.to_string(), "1".to_string());
        }
        self.submit(sh, sid, local, &spec, idx, ts, &format!("close-{n}"), None).map(|_| ())
    }

    // -- reports ---------------------------------------------------------

    /// Hear what a simulated venue reported in answer to orders sent after the step's poll (an
    /// answer to a fill, met on the bar of the step): they are booked and heard in the same step.
    pub(super) fn hear_same_event(
        &self,
        py: Python<'_>,
        sh: &Shared,
        st: &Strat,
        ev: &Ev<'_, '_>,
        local: usize,
    ) -> PyResult<bool> {
        if !self.simulated {
            return Ok(false);
        }
        let mut out: std::collections::VecDeque<_> = {
            let mut adapter = self.adapter.borrow_mut();
            adapter
                .as_simulation()
                .expect("checked when the route was built")
                .take_same_event()
                .into()
        };
        let mut heard = false;
        while let Some((venue_id, report)) = out.pop_front() {
            sh.jlog_now(jk::SIMULATED_REPORT, jsrc::ENGINE, Some(&venue_id), || {
                super::report_value(&report)
            })?;
            heard |= self.deliver(
                py,
                sh,
                st,
                &venue_id,
                &report,
                ev.event.local_idx as usize,
                ev.event.ts,
            )? != ReportOutcome::Duplicate;
            let mut adapter = self.adapter.borrow_mut();
            out.extend(adapter.as_simulation().expect("checked").take_same_event());
        }
        let _ = local;
        Ok(heard)
    }

    /// A simulated venue's turn, first half: step it with the event the strategy's book has just
    /// stepped on and collect what it reports. Nothing is told to the strategy yet
    /// ([`VenueRoute::poll_deliver`]), so every instrument of a round is matched before a hook
    /// hears any fill.
    pub(super) fn poll_step(
        &self,
        sh: &Shared,
        st: &Strat,
        ev: &Ev<'_, '_>,
        local: usize,
        atr: f64,
    ) -> PyResult<Option<VenueReports>> {
        if !self.simulated {
            return Ok(None);
        }
        let (entry, depth) = match &*sh.inner.borrow() {
            Inner::Running(c) => {
                let cur = c.current().ok_or_else(finished)?;
                (
                    ScheduleEntry {
                        instrument: local,
                        local_idx: ev.event.local_idx as usize,
                        data: cur.data,
                    },
                    c.depth_of(&cur.event),
                )
            }
            _ => return Err(finished()),
        };
        let mut out = Vec::new();
        let mut adapter = self.adapter.borrow_mut();
        let venue = adapter.as_simulation().expect("checked when the route was built");
        // The shadow book moves a position only on an order: a change it makes on its own is a
        // fault of the engine, not of the caller's input, so it stops the run as an internal error.
        venue.on_market_event(st.sid.index(), &entry, depth, atr, &mut out).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("internal error (paper venue): {e}"))
        })?;
        out.extend(venue.take_same_event());
        Ok(Some(out))
    }

    /// A simulated venue's turn at the open of a bar, before the strategy has seen it
    /// (`next_bar_open`): what was decided earlier fills at the open, and the venue reports it.
    /// [`VenueRoute::poll_step`] carries on with the rest of the bar. `None` when there is
    /// nothing to report.
    pub(super) fn poll_open(
        &self,
        sh: &Shared,
        st: &Strat,
        ev: &Ev<'_, '_>,
        local: usize,
    ) -> PyResult<Option<VenueReports>> {
        if !self.simulated {
            return Ok(None);
        }
        let entry = match &*sh.inner.borrow() {
            Inner::Running(c) => {
                let cur = c.current().ok_or_else(finished)?;
                ScheduleEntry {
                    instrument: local,
                    local_idx: ev.event.local_idx as usize,
                    data: cur.data,
                }
            }
            _ => return Err(finished()),
        };
        let mut out = Vec::new();
        let mut adapter = self.adapter.borrow_mut();
        let venue = adapter.as_simulation().expect("checked when the route was built");
        venue.on_open(st.sid.index(), &entry, &mut out).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("internal error (paper venue): {e}"))
        })?;
        out.extend(venue.take_same_event());
        Ok((!out.is_empty()).then_some(out))
    }

    /// A simulated venue's turn while the bar forms, before the strategy has decided on it: its
    /// stops, targets and resting orders trade the bar's range and the venue reports it.
    /// [`VenueRoute::poll_step`] carries on with the decisions of the bar. `None` when there is
    /// nothing to report.
    pub(super) fn poll_rest(
        &self,
        sh: &Shared,
        st: &Strat,
        ev: &Ev<'_, '_>,
        local: usize,
    ) -> PyResult<Option<VenueReports>> {
        if !self.simulated {
            return Ok(None);
        }
        let entry = match &*sh.inner.borrow() {
            Inner::Running(c) => {
                let cur = c.current().ok_or_else(finished)?;
                ScheduleEntry {
                    instrument: local,
                    local_idx: ev.event.local_idx as usize,
                    data: cur.data,
                }
            }
            _ => return Err(finished()),
        };
        let mut out = Vec::new();
        let mut adapter = self.adapter.borrow_mut();
        let venue = adapter.as_simulation().expect("checked when the route was built");
        venue.on_rest(st.sid.index(), &entry, &mut out).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("internal error (paper venue): {e}"))
        })?;
        out.extend(venue.take_same_event());
        Ok((!out.is_empty()).then_some(out))
    }

    /// Book what the venue reported for the event now, keeping the events for the hooks:
    /// [`VenueRoute::hear_held`] tells the strategy once the whole round is matched.
    pub(super) fn poll_book(
        &self,
        py: Python<'_>,
        sh: &Shared,
        st: &Strat,
        ev: &Ev<'_, '_>,
        out: VenueReports,
    ) -> PyResult<(bool, HeldEvents)> {
        self.hold.set(true);
        let heard = self.poll_deliver(py, sh, st, ev, out);
        self.hold.set(false);
        let held = std::mem::take(&mut *self.held.borrow_mut());
        Ok((heard?, held))
    }

    /// [`VenueRoute::poll_book`] for what a simulated venue reported while the bar formed, before
    /// the strategy decides on it: a protective leg that lapsed with the last session is sent
    /// again here, and meets this bar as the backtest's level does, so its fill is booked before
    /// the strategy's hooks for the bar run, as the batch's is.
    pub(super) fn poll_book_rest(
        &self,
        py: Python<'_>,
        sh: &Shared,
        st: &Strat,
        ev: &Ev<'_, '_>,
        out: VenueReports,
        local: usize,
    ) -> PyResult<(bool, HeldEvents)> {
        self.hold.set(true);
        let heard = (|| -> PyResult<bool> {
            let mut heard = self.poll_deliver(py, sh, st, ev, out)?;
            self.rearm(sh, st.sid, local, ev.event.local_idx as usize, ev.event.ts)?;
            let more = {
                let mut adapter = self.adapter.borrow_mut();
                adapter.as_simulation().map(|v| v.take_same_event()).unwrap_or_default()
            };
            if !more.is_empty() {
                heard |= self.poll_deliver(py, sh, st, ev, more)?;
            }
            Ok(heard)
        })();
        self.hold.set(false);
        let held = std::mem::take(&mut *self.held.borrow_mut());
        Ok((heard?, held))
    }

    /// Tell the strategy the events a report made, which were booked earlier in the step.
    pub(super) fn hear_held(
        &self,
        py: Python<'_>,
        sh: &Shared,
        st: &Strat,
        held: HeldEvents,
    ) -> PyResult<()> {
        for (local, events) in held {
            let ctx = st.ctx.bind(py);
            let before = ctx.borrow().local.replace(local);
            let global = sh.global_of(st.sid, local)?;
            let symbol = sh.symbol(py, global.index());
            let dispatched = super::heard(py, sh, st, events, &mut |events| {
                dispatch_events(
                    py,
                    st.strategy.bind(py),
                    ctx.as_any(),
                    live_mask(py, &st.queue, st.mask),
                    events,
                    Some(&symbol),
                )
            });
            ctx.borrow().local.set(before);
            dispatched?;
        }
        Ok(())
    }

    /// The second half: book what the venue reported and tell the strategy. Returns whether it
    /// heard of anything.
    pub(super) fn poll_deliver(
        &self,
        py: Python<'_>,
        sh: &Shared,
        st: &Strat,
        ev: &Ev<'_, '_>,
        out: VenueReports,
    ) -> PyResult<bool> {
        let local_idx = ev.event.local_idx as usize;
        let mut heard = false;
        let mut out: std::collections::VecDeque<_> = out.into();
        while let Some((venue_id, report)) = out.pop_front() {
            // The simulated venue's answer is its own decision, not an input
            // the host pushed: it is on the record, with its reason, before
            // the book takes it.
            sh.jlog_now(jk::SIMULATED_REPORT, jsrc::ENGINE, Some(&venue_id), || {
                super::report_value(&report)
            })?;
            heard |= self.deliver(py, sh, st, &venue_id, &report, local_idx, ev.event.ts)?
                != ReportOutcome::Duplicate;
            // Booking a fill sends the position's protective legs, which meet
            // the rest of this same bar: their reports belong to this event.
            let mut adapter = self.adapter.borrow_mut();
            let venue = adapter.as_simulation().expect("checked when the route was built");
            out.extend(venue.take_same_event());
        }
        Ok(heard)
    }

    /// Fold a report into the book of the strategy it belongs to and tell
    /// the strategy what it did. The one way a report reaches a book.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn deliver(
        &self,
        py: Python<'_>,
        sh: &Shared,
        st: &Strat,
        venue_id: &str,
        report: &ExecutionReport,
        idx: usize,
        ts: i64,
    ) -> PyResult<ReportOutcome> {
        let Some(placed) = self.placed(venue_id) else {
            // An order dropped after it ended: whatever the venue says about
            // it now no longer matters.
            if self.is_dropped(venue_id) {
                return Ok(ReportOutcome::Late);
            }
            return Err(Fail::new(
                "unknown_client_order_id",
                format!("the venue reported on order {venue_id:?}, which was never sent"),
            )
            .param("client_order_id")
            .value(venue_id)
            .into());
        };
        if placed.sid != st.sid.index() {
            return Err(Fail::new("unknown_client_order_id", "the report is for another strategy")
                .param("client_order_id")
                .value(venue_id)
                .into());
        }
        // A venue that amends an order on its own (a simulated one sizing a
        // capital-fraction order as it fills) is recorded as the change asked
        // and answered.
        if matches!(report, ExecutionReport::ReplaceAck { .. }) {
            let pending = sh.with_book(st.sid, |b, _| {
                Ok(b.kernel(placed.local)
                    .order(placed.order_id)
                    .is_some_and(|o| o.status == OrderStatus::PendingReplace))
            })?;
            if !pending {
                // Not a working order any more: the acknowledgement is late.
                let _ = sh.with_book_mut(st.sid, |b, _| {
                    Ok(b.request_venue_replace(placed.local, placed.order_id))
                })?;
            }
        }
        // What the order was before the report, to tell a leg that lapsed with its session from
        // one the engine or the strategy took off.
        let status_before = sh.with_book(st.sid, |b, _| {
            Ok(b.kernel(placed.local).order(placed.order_id).map(|o| o.status))
        })?;
        let halted_before = sh.halted(st.sid);
        let (outcome, events) = sh.with_book_mut(st.sid, |book, _| {
            book.apply_venue_report(placed.local, idx, ts, placed.order_id, report)
                .map_err(venue_err)
        })?;
        sh.journal_halt_edge(st.sid, halted_before, &events)?;
        if matches!(report, ExecutionReport::Fill { .. })
            && outcome == ReportOutcome::Applied
            && sh.with_book(st.sid, |b, _| {
                Ok(b.kernel(placed.local)
                    .order(placed.order_id)
                    .is_some_and(|o| o.status == OrderStatus::Filled))
            })?
        {
            self.emulate_oco(sh, st.sid, venue_id, idx)?;
        }
        super::journal_engine_events(sh, st, &events, true)?;
        // A fill the book could not book yet is on the record, so recovery
        // knows the book and the venue disagree until it is booked.
        for event in &events {
            if let crate::portfolio::EngineEvent::FillDeferred {
                order_id,
                exec_id,
                price,
                size,
                reason,
                ..
            } = event
            {
                sh.jlog_now(jk::UNBOOKED_FILL, jsrc::ENGINE, Some(venue_id), || {
                    JV::map([
                        ("order_id", JV::Int(*order_id as i64)),
                        ("exec_id", JV::str(exec_id)),
                        ("price", JV::Float(*price)),
                        ("qty", JV::Float(*size)),
                        ("reason", JV::str(reason)),
                    ])
                })?;
            }
        }
        self.after_booking(sh, st.sid, placed.local, idx, ts)?;
        self.note_if_ended(sh, st.sid, placed.local, placed.order_id)?;
        if let Some(before) = status_before {
            let report_ts = crate::python::execution_bindings::report_ts(report);
            self.note_lapse(sh, st.sid, &placed, venue_id, before, report_ts, idx, ts)?;
            self.note_rearm_end(sh, st.sid, &placed, venue_id, before, report, idx, ts)?;
        }
        if !events.is_empty() && self.hold.get() {
            self.held.borrow_mut().push((placed.local, events));
        } else if !events.is_empty() {
            let ctx = st.ctx.bind(py);
            let before = ctx.borrow().local.replace(placed.local);
            let global = sh.global_of(st.sid, placed.local)?;
            let symbol = sh.symbol(py, global.index());
            let dispatched = super::heard(py, sh, st, events, &mut |events| {
                dispatch_events(
                    py,
                    st.strategy.bind(py),
                    ctx.as_any(),
                    live_mask(py, &st.queue, st.mask),
                    events,
                    Some(&symbol),
                )
            });
            ctx.borrow().local.set(before);
            dispatched?;
        }
        Ok(outcome)
    }

    // -- what a backtest does in-process, as orders ----------------------

    fn next_prot(&self, sid: StrategyId) -> u64 {
        let mut ledger = self.ledger.borrow_mut();
        let n = ledger.prot[sid.index()];
        ledger.prot[sid.index()] += 1;
        n
    }

    fn spec(side: OrderSide, kind: &str, units: Option<f64>, reduce_only: bool) -> OrderSpec {
        OrderSpec {
            side: side.as_str().to_string(),
            kind: kind.to_string(),
            units,
            size_frac: None,
            limit_price: None,
            trigger_price: None,
            tif: TIF_ENGINE.to_string(),
            expire_ns: None,
            stop_price: None,
            target_price: None,
            offset: None,
            offset_kind: "price".to_string(),
            limit_offset: 0.0,
            post_only: false,
            reduce_only,
            position_id: None,
            tags: Default::default(),
        }
    }

    /// Orders made out of the strategy's `enter()` and `close_position()`
    /// intents on this event.
    pub(super) fn intents(
        &self,
        sh: &Shared,
        st: &Strat,
        local: usize,
        input: &StepInput,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        let sid = st.sid;
        if input.exit {
            let working_close = sh.with_book(sid, |b, _| {
                Ok(b.kernel(local).open_orders().iter().any(|o| o.client_id.starts_with("close-")))
            })?;
            if !working_close {
                let (ids, entering): (Vec<u64>, bool) = sh.with_book(sid, |b, _| {
                    let k = b.kernel(local);
                    Ok((
                        k.position_snapshots().iter().map(|p| p.position_id).collect(),
                        k.open_orders().iter().any(|o| o.client_id.starts_with("enter-")),
                    ))
                })?;
                if ids.is_empty() && entering {
                    self.ledger.borrow_mut().held_exits.insert((sid.index(), local));
                }
                for id in ids {
                    self.close(sh, sid, local, idx, ts, id, false)?;
                }
            }
        }
        if input.entry {
            let (held, pending, direction) = sh.with_book(sid, |b, _| {
                let k = b.kernel(local);
                Ok((
                    !k.position_snapshots().is_empty(),
                    k.open_orders().iter().any(|o| o.client_id.starts_with("enter-")),
                    k.signal_direction(),
                ))
            })?;
            if held || pending {
                return Ok(());
            }
            let side = if direction == crate::core::types::Direction::Long {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            };
            use crate::portfolio::risk::RejectReason;
            let sized = match sh.current_bar(sid, local) {
                Some(bar) => sh.with_book_mut(sid, |b, _| {
                    Ok(b.venue_units(
                        local,
                        &bar,
                        side,
                        Some(input.size_mult.unwrap_or(1.0)),
                        None,
                        false,
                    ))
                })?,
                None => Err("zero_size"),
            };
            if let Err(why) = sized {
                // Sized to nothing: the entry is refused as it is in-process.
                let reason = match why {
                    "insufficient_margin" => RejectReason::InsufficientMargin,
                    "insufficient_capital" => RejectReason::InsufficientCapital,
                    _ => RejectReason::ZeroSize,
                };
                return sh.with_book_mut(sid, |b, _| {
                    b.kernel_mut(local)
                        .pending_events
                        .push(crate::portfolio::kernel::EngineEvent::EntryRejected { idx, reason });
                    Ok(())
                });
            }
            // The entry is a share of the capital, sized when it fills (as a
            // capital-fraction order is); the check above only decided that
            // there is something to trade. At a venue the quantity sent is
            // the estimate of that moment.
            let mut spec = Self::spec(side, "market", None, false);
            spec.size_frac = Some(input.size_mult.unwrap_or(1.0));
            spec.stop_price = input.stop_price_override;
            spec.target_price = input.target_price_override;
            let n = self.next_prot(sid);
            self.submit(sh, sid, local, &spec, idx, ts, &format!("enter-{n}"), None)?;
        }
        Ok(())
    }

    /// After the strategy's book has stepped with an event: send what the
    /// kernel left for the venue (schedule slices, the square-off) and answer
    /// any level the engine is watching.
    pub(super) fn flush(
        &self,
        sh: &Shared,
        st: &Strat,
        local: usize,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        let sid = st.sid;
        let (slices, squareoff) = sh.with_book_mut(sid, |b, _| {
            let k = b.kernel_mut(local);
            Ok((k.take_slices(), k.take_squareoff()))
        })?;
        for slice in slices {
            let spec = Self::spec(slice.side, "market", Some(slice.units), slice.reduce_only);
            self.submit(sh, sid, local, &spec, idx, ts, &slice.client_id, None)?;
        }
        if squareoff {
            sh.jlog_now(jk::ENGINE_EXIT, jsrc::ENGINE, None, || {
                JV::map([
                    ("strategy", JV::Int(sid.index() as i64)),
                    ("reason", JV::str("squareoff")),
                    ("venue", JV::Bool(true)),
                ])
            })?;
            self.cancel_adding(sh, sid, local, idx)?;
            // A hedge leg's square-off closes the sold legs of its group first.
            self.flatten(sh, sid, &[(local, idx)], ts, true)?;
        }
        // A protective leg that lapsed with the last session is sent again on the first event of
        // this one, after the strategy's decisions and the square-off: a close they sent holds it.
        self.rearm(sh, sid, local, idx, ts)?;
        self.watch(sh, sid, local, idx, ts)
    }

    /// Answer the levels the engine watches that the event reached.
    fn watch(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        if self.ledger.borrow().watchers.is_empty() {
            return Ok(());
        }
        let Some(bar) = sh.current_bar(sid, local) else { return Ok(()) };
        let mut hit = Vec::new();
        {
            let mut ledger = self.ledger.borrow_mut();
            ledger.watchers.retain_mut(|w| {
                if w.sid != sid.index() || w.local != local {
                    return true;
                }
                // A long's exit is a sell: its stop is below, its target above.
                let sell = w.exit == OrderSide::Sell;
                let stop = w.stop.is_some_and(|p| if sell { bar.low <= p } else { bar.high >= p });
                let target =
                    w.target.is_some_and(|p| if sell { bar.high >= p } else { bar.low <= p });
                if stop || target {
                    let level = if stop { w.stop } else { w.target }.unwrap_or(bar.open);
                    // The bar opened through the level: nothing traded between the open and it.
                    let through = if sell == stop { bar.open <= level } else { bar.open >= level };
                    hit.push((
                        w.exit,
                        w.qty,
                        w.position_id,
                        if stop { "watch_stop" } else { "watch_target" },
                        through,
                    ));
                    false
                } else {
                    // A trailing level stands where a backtest's would after this bar: below the
                    // highest price since entry, never lower than it was.
                    if let Some(t) = w.trail.as_mut() {
                        t.high = t.high.max(bar.high);
                        let level = t.high * (1.0 - t.fraction);
                        if w.stop.is_none_or(|p| level > p) {
                            w.stop = Some(level);
                        }
                    }
                    true
                }
            });
        }
        for (exit, qty, position_id, why, opened_through) in hit {
            // Under `next_bar_open` a level touched after the open is answered at the next bar's
            // open; one the bar opened through is answered at this one.
            let answers_open = !sh
                .with_book(sid, |b, _| Ok(b.kernel(local).fills_at_next_open()))?
                || opened_through;
            sh.jlog_now(jk::ENGINE_EXIT, jsrc::ENGINE, None, || {
                JV::map([
                    ("strategy", JV::Int(sid.index() as i64)),
                    ("reason", JV::str(why)),
                    ("size", JV::Float(qty)),
                    ("venue", JV::Bool(true)),
                ])
            })?;
            if let Some(position) = self.hedge_to_cover(sh, sid, local, position_id)? {
                self.cover_hedge(sh, sid, local, position, idx, ts, answers_open)?;
                continue;
            }
            let n = self.next_prot(sid);
            let mut spec = Self::spec(exit, "market", Some(qty), true);
            spec.position_id = position_id;
            self.submit(sh, sid, local, &spec, idx, ts, &format!("prot-{n}"), None)?;
        }
        Ok(())
    }

    /// The position to close as a hedge leg, when the instrument's level is the engine's (a
    /// bought option of a margin group): `position_id`, or the one position held when the account
    /// nets.
    fn hedge_to_cover(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        position_id: Option<u64>,
    ) -> PyResult<Option<u64>> {
        sh.with_book(sid, |b, _| {
            let k = b.kernel(local);
            if !k.engine_holds_levels(crate::core::types::Direction::Long) {
                return Ok(None);
            }
            Ok(k.position_snapshots()
                .iter()
                .find(|p| {
                    p.direction == crate::core::types::Direction::Long
                        && position_id.is_none_or(|id| id == p.position_id)
                })
                .map(|p| p.position_id)
                // Only a hedge whose close leaves a sold leg without cover is the engine's to
                // close; any other level is an ordinary order.
                .filter(|&id| !b.sold_legs_left_bare(local, id).is_empty()))
        })
    }

    /// Close a hedge leg the way the engine holds it: reduce-only market closes of the sold legs
    /// of its group first, the hedge's own once none is left (see `release_covers`). Each is an
    /// ordinary order, sent and heard like any other.
    #[allow(clippy::too_many_arguments)]
    fn cover_hedge(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        position_id: u64,
        idx: usize,
        ts: i64,
        follow_up: bool,
    ) -> PyResult<()> {
        let sold = sh.with_book(sid, |b, _| Ok(b.sold_legs_left_bare(local, position_id)))?;
        let legs: Vec<(usize, u64)> = sold.iter().map(|&(j, id, _)| (j, id)).collect();
        for (j, id, size) in sold {
            if self.close_working(sh, sid, j, id)? {
                continue;
            }
            let hedging = sh.with_book(sid, |b, _| Ok(b.kernel(j).is_hedging()))?;
            let mut spec = Self::spec(OrderSide::Buy, "market", Some(size), true);
            spec.position_id = if hedging { Some(id) } else { None };
            if follow_up {
                spec.tags.insert(super::FOLLOW_UP_TAG.to_string(), "1".to_string());
            }
            let n = self.next_prot(sid);
            self.submit(sh, sid, j, &spec, idx, ts, &format!("prot-{n}"), None)?;
        }
        let key = (sid.index(), local, position_id);
        let queued =
            self.ledger.borrow().covers.iter().any(|c| (c.sid, c.local, c.position_id) == key);
        if !queued {
            self.ledger.borrow_mut().covers.push(HeldCover {
                sid: key.0,
                local,
                position_id,
                sold: legs,
            });
        }
        self.release_covers(sh, sid, idx, ts, follow_up)
    }

    /// Send the close of every held hedge leg whose group has no sold leg left. `follow_up` says
    /// whether the close meets the bar of the event it answers (the open under `next_bar_open`,
    /// when the level was touched at it); a close sent after the sold legs' own fills always does.
    fn release_covers(
        &self,
        sh: &Shared,
        sid: StrategyId,
        idx: usize,
        ts: i64,
        follow_up: bool,
    ) -> PyResult<()> {
        let held: Vec<HeldCover> =
            self.ledger.borrow().covers.iter().filter(|c| c.sid == sid.index()).cloned().collect();
        for cover in held {
            let (sold_left, open) = sh.with_book(sid, |b, _| {
                Ok((
                    cover.sold.iter().any(|&(j, id)| {
                        b.kernel(j).position_snapshots().iter().any(|p| p.position_id == id)
                    }),
                    b.kernel(cover.local)
                        .position_snapshots()
                        .iter()
                        .any(|p| p.position_id == cover.position_id),
                ))
            })?;
            if open && sold_left {
                continue;
            }
            self.ledger.borrow_mut().covers.retain(|c| {
                (c.sid, c.local, c.position_id) != (cover.sid, cover.local, cover.position_id)
            });
            if open {
                self.close(sh, sid, cover.local, idx, ts, cover.position_id, follow_up)?;
            }
        }
        Ok(())
    }

    /// Keep each hedge leg's stop and target where its group needs them. A level is the engine's
    /// while closing the leg would leave a sold leg of its margin group without cover: the venue
    /// holds no order for it, so a level that trips closes the sold legs left bare first
    /// ([`VenueRoute::cover_hedge`]). Any other level is an ordinary resting order at the venue.
    /// What a group needs changes as its legs open and close, so the levels move between the two
    /// with it: the venue's orders are cancelled and the levels watched when a sold leg comes to
    /// depend on a hedge, and sent back as orders once none does.
    fn reconcile_hedge_levels(
        &self,
        sh: &Shared,
        sid: StrategyId,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        if self.protective_local {
            return Ok(());
        }
        // (local instrument, position id, size, whether a sold leg depends on it)
        let legs: Vec<(usize, u64, f64, bool)> = sh.with_book(sid, |b, _| {
            let mut out = Vec::new();
            for m in 0..b.kernel_count() {
                let k = b.kernel(m);
                if !k.engine_holds_levels(crate::core::types::Direction::Long) {
                    continue;
                }
                for p in k.position_snapshots() {
                    if p.direction == crate::core::types::Direction::Long {
                        let needed = !b.sold_legs_left_bare(m, p.position_id).is_empty();
                        out.push((m, p.position_id, p.size, needed));
                    }
                }
            }
            Ok(out)
        })?;
        for (m, position_id, size, needed) in legs {
            let hedging = sh.with_book(sid, |b, _| Ok(b.kernel(m).is_hedging()))?;
            let named = if hedging { Some(position_id) } else { None };
            let watched = self.ledger.borrow().watchers.iter().any(|w| {
                w.sid == sid.index()
                    && w.local == m
                    && w.exit == OrderSide::Sell
                    && w.position_id == named
            });
            if needed && !watched {
                self.take_levels(sh, sid, m, position_id, OrderSide::Sell, size, named, idx)?;
            } else if !needed && watched {
                let watch = {
                    let mut ledger = self.ledger.borrow_mut();
                    let at = ledger.watchers.iter().position(|w| {
                        w.sid == sid.index()
                            && w.local == m
                            && w.exit == OrderSide::Sell
                            && w.position_id == named
                    });
                    at.map(|at| ledger.watchers.remove(at))
                };
                if let Some(w) = watch {
                    let request = crate::portfolio::kernel::ProtectRequest {
                        direction: crate::core::types::Direction::Long,
                        size: w.qty,
                        entry_price: w.trail.map_or(0.0, |t| t.entry),
                        stop: w.stop,
                        target: w.target,
                        trail: w.trail.map(|t| t.entry * t.fraction),
                        position_id,
                        amend: false,
                    };
                    self.protect(sh, sid, m, &request, idx, ts)?;
                }
            }
        }
        Ok(())
    }

    /// The venue's resting stop and target of a hedge leg become the engine's: the orders are
    /// cancelled and their levels watched.
    #[allow(clippy::too_many_arguments)]
    fn take_levels(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        position_id: u64,
        exit: OrderSide,
        size: f64,
        named: Option<u64>,
        idx: usize,
    ) -> PyResult<()> {
        let orders: Vec<(u64, Option<u64>, crate::execution::orders::OrderKind, Option<f64>)> = sh
            .with_book(sid, |b, _| {
                Ok(b.kernel(local)
                    .open_orders()
                    .iter()
                    .filter(|o| {
                        o.client_id.starts_with("prot-")
                            && o.reduce_only
                            && o.status != OrderStatus::PendingCancel
                    })
                    .map(|o| (o.id, o.position_id, o.kind, o.trail_watermark))
                    .collect())
            })?;
        let entry = sh.with_book(sid, |b, _| {
            Ok(b.kernel(local)
                .position_snapshots()
                .iter()
                .find(|p| p.position_id == position_id)
                .map(|p| p.entry_price))
        })?;
        let mut watch = Watch {
            sid: sid.index(),
            local,
            exit,
            qty: size,
            stop: None,
            target: None,
            position_id: named,
            trail: None,
        };
        let mut taken = Vec::new();
        for (order_id, order_position, kind, watermark) in orders {
            if order_position.is_some_and(|p| p != position_id) {
                continue;
            }
            match kind {
                crate::execution::orders::OrderKind::StopMarket { trigger } => {
                    watch.stop = Some(trigger)
                }
                crate::execution::orders::OrderKind::Limit { price } => watch.target = Some(price),
                // The trailing stop of a hedge the engine carries trails as a backtest's does:
                // the offset sent with the order is the entry price times the stop's fraction.
                crate::execution::orders::OrderKind::TrailingStopMarket { offset }
                    if entry.is_some_and(|e| e > 0.0) =>
                {
                    let entry = entry.unwrap_or_default();
                    let fraction = match offset {
                        crate::execution::orders::TrailOffset::Price(price) => price / entry,
                        crate::execution::orders::TrailOffset::Bps(bps) => bps / 10_000.0,
                    };
                    let high = watermark.unwrap_or(entry).max(entry);
                    watch.stop = Some(high * (1.0 - fraction));
                    watch.trail = Some(Trail { fraction, entry, high });
                }
                // A trailing level the engine cannot watch stays with the venue.
                _ => continue,
            }
            taken.push(order_id);
        }
        if taken.is_empty() {
            return Ok(());
        }
        for order_id in taken {
            self.cancel(sh, sid, local, idx, order_id)?;
        }
        self.ledger.borrow_mut().watchers.push(watch);
        Ok(())
    }

    /// Protection for the positions opened, and none for a flat instrument.
    fn after_booking(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        self.release_covers(sh, sid, idx, ts, true)?;
        self.reconcile_hedge_levels(sh, sid, idx, ts)?;
        let (requests, flat) = sh.with_book_mut(sid, |b, _| {
            let k = b.kernel_mut(local);
            Ok((k.take_protect_requests(), k.position_snapshots().is_empty()))
        })?;
        for r in &requests {
            self.protect(sh, sid, local, r, idx, ts)?;
        }
        self.resize_protection(sh, sid, local, idx)?;
        self.extend_protection(sh, sid, local, idx, ts, &requests)?;
        // Under hedging a position's protection ends with that position, while others stay
        // open: cancel the protective orders (and drop the watched levels) of positions that
        // are gone, so none is left to be refused for a position it names.
        let (live, orders): (Vec<u64>, Vec<(u64, u64)>) = sh.with_book(sid, |b, _| {
            let k = b.kernel(local);
            Ok((
                k.position_snapshots().iter().map(|p| p.position_id).collect(),
                k.open_orders()
                    .iter()
                    .filter(|o| {
                        o.client_id.starts_with("prot-") && o.status != OrderStatus::PendingCancel
                    })
                    .filter_map(|o| o.position_id.map(|p| (o.id, p)))
                    .collect(),
            ))
        })?;
        self.ledger.borrow_mut().watchers.retain(|w| {
            !(w.sid == sid.index()
                && w.local == local
                && w.position_id.is_some_and(|p| !live.contains(&p)))
        });
        for (order_id, position_id) in orders {
            if !live.contains(&position_id) {
                self.cancel(sh, sid, local, idx, order_id)?;
            }
        }
        let held_exit = self.ledger.borrow().held_exits.contains(&(sid.index(), local));
        if held_exit {
            if !flat {
                self.ledger.borrow_mut().held_exits.remove(&(sid.index(), local));
                let ids: Vec<u64> = sh.with_book(sid, |b, _| {
                    Ok(b.kernel(local).position_snapshots().iter().map(|p| p.position_id).collect())
                })?;
                for id in ids {
                    self.close(sh, sid, local, idx, ts, id, false)?;
                }
            } else {
                let entering = sh.with_book(sid, |b, _| {
                    Ok(b.kernel(local)
                        .open_orders()
                        .iter()
                        .any(|o| o.client_id.starts_with("enter-")))
                })?;
                if !entering {
                    self.ledger.borrow_mut().held_exits.remove(&(sid.index(), local));
                }
            }
        }
        if flat && requests.is_empty() {
            self.ledger
                .borrow_mut()
                .watchers
                .retain(|w| !(w.sid == sid.index() && w.local == local));
            let ids: Vec<u64> = sh.with_book(sid, |b, _| {
                Ok(b.kernel(local)
                    .open_orders()
                    .iter()
                    .filter(|o| {
                        o.client_id.starts_with("prot-") && o.status != OrderStatus::PendingCancel
                    })
                    .map(|o| o.id)
                    .collect())
            })?;
            for id in ids {
                self.cancel(sh, sid, local, idx, id)?;
            }
        }
        Ok(())
    }

    /// Where resting protection counts against the line, an add is protected by orders of its
    /// own: what the position holds beyond what its resting stop and target cover gets a stop and
    /// a target of the same levels. (The venue sizes a leg to the order it hangs on, so a leg is
    /// never grown to take in an add.) Only a netting position: under hedging an add is a
    /// position with its own protection.
    fn extend_protection(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        ts: i64,
        requests: &[crate::portfolio::kernel::ProtectRequest],
    ) -> PyResult<()> {
        if !self.exits_hold_line || self.protective_local {
            return Ok(());
        }
        // (direction, size) of the position, the levels and the quantity covered by each
        type Cover = (crate::core::types::Direction, f64, Option<(f64, f64)>, Option<(f64, f64)>);
        let cover: Option<Cover> = sh.with_book(sid, |b, _| {
            let k = b.kernel(local);
            if k.is_hedging() {
                return Ok(None);
            }
            let Some(p) = k.position_snapshots().into_iter().next() else { return Ok(None) };
            let mut stop: Option<(f64, f64)> = None;
            let mut target: Option<(f64, f64)> = None;
            for o in k.open_orders() {
                if !o.client_id.starts_with("prot-")
                    || !o.reduce_only
                    || matches!(o.status, OrderStatus::PendingCancel)
                {
                    continue;
                }
                let crate::execution::orders::QtySpec::Units(q) = o.qty else { continue };
                let left = (q - o.filled_qty).max(0.0);
                match o.kind {
                    crate::execution::orders::OrderKind::StopMarket { trigger } => {
                        stop = Some((trigger, stop.map_or(0.0, |s| s.1) + left));
                    }
                    crate::execution::orders::OrderKind::Limit { price } => {
                        target = Some((price, target.map_or(0.0, |s| s.1) + left));
                    }
                    _ => {}
                }
            }
            Ok(Some((p.direction, p.size, stop, target)))
        })?;
        let Some((direction, size, stop, target)) = cover else { return Ok(()) };
        let exit = if direction == crate::core::types::Direction::Long {
            OrderSide::Sell
        } else {
            OrderSide::Buy
        };
        // A level an add just moved is still on the old order until the venue acknowledges the
        // change: the bare quantity is placed at the level the add named.
        let named = |stop: bool| {
            requests
                .iter()
                .rev()
                .filter(|r| r.amend)
                .find_map(|r| if stop { r.stop } else { r.target })
        };
        let mut ids = Vec::new();
        for (level, covered, is_stop) in [(stop, true), (target, false)]
            .into_iter()
            .filter_map(|(l, k)| l.map(|(level, covered)| (named(k).unwrap_or(level), covered, k)))
        {
            let bare = size - covered;
            if bare <= 1e-9 {
                continue;
            }
            if let Some(id) =
                self.place_leg(sh, sid, local, idx, ts, exit, is_stop, bare, level, None)?
            {
                ids.push(id);
            }
        }
        if ids.len() == 2 {
            self.link_oco(sid, local, &ids)?;
        }
        Ok(())
    }

    /// Send one reduce-only protective leg of `qty` at `level`; the id of the order when the venue
    /// holds it. A venue that cannot hold it leaves the level to the engine, which watches it.
    #[allow(clippy::too_many_arguments)]
    fn place_leg(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        ts: i64,
        exit: OrderSide,
        is_stop: bool,
        qty: f64,
        level: f64,
        position_id: Option<u64>,
    ) -> PyResult<Option<u64>> {
        self.place_leg_tagged(
            sh,
            sid,
            local,
            idx,
            ts,
            exit,
            is_stop,
            qty,
            level,
            position_id,
            false,
        )
    }

    /// [`VenueRoute::place_leg`]; `rearm` marks a leg sent again after its day order lapsed,
    /// which a simulated venue meets with the bar of the event it is sent on.
    #[allow(clippy::too_many_arguments)]
    fn place_leg_tagged(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
        ts: i64,
        exit: OrderSide,
        is_stop: bool,
        qty: f64,
        level: f64,
        position_id: Option<u64>,
        rearm: bool,
    ) -> PyResult<Option<u64>> {
        let mut spec =
            Self::spec(exit, if is_stop { "stop_market" } else { "limit" }, Some(qty), true);
        if is_stop {
            spec.trigger_price = Some(level);
        } else {
            spec.limit_price = Some(level);
        }
        spec.position_id = position_id;
        if rearm {
            spec.tags.insert(crate::execution::adapter::REARM_TAG.to_string(), "1".to_string());
        }
        let n = self.next_prot(sid);
        self.unsupported_sent.set(false);
        let id = self.submit(sh, sid, local, &spec, idx, ts, &format!("prot-{n}"), None)?;
        if !self.unsupported_sent.replace(false) {
            return Ok(Some(id));
        }
        let mut watch = Watch {
            sid: sid.index(),
            local,
            exit,
            qty,
            stop: None,
            target: None,
            position_id,
            trail: None,
        };
        if is_stop {
            watch.stop = Some(level);
        } else {
            watch.target = Some(level);
        }
        self.ledger.borrow_mut().watchers.push(watch);
        Ok(None)
    }

    /// The protective orders and watched levels cover what the position holds now. An add or a
    /// partial close changes the size after they were placed for it; left at the placement size
    /// an add is half protected, and the stop of a position part closed is oversized.
    fn resize_protection(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
    ) -> PyResult<()> {
        // (order id, position it names, its quantity, what it has filled) and (position id, size,
        // whether hedging)
        type Held = (Vec<(u64, Option<u64>, f64, f64)>, Vec<(u64, f64)>, bool);
        let (orders, positions, hedging): Held = sh.with_book(sid, |b, _| {
            let k = b.kernel(local);
            let orders = k
                .open_orders()
                .iter()
                .filter(|o| {
                    // An order the venue has not accepted cannot be changed yet; its
                    // acceptance is booked through here again.
                    o.client_id.starts_with("prot-")
                        && !matches!(
                            o.status,
                            OrderStatus::Submitted
                                | OrderStatus::PendingCancel
                                | OrderStatus::PendingReplace
                        )
                })
                .filter_map(|o| match o.qty {
                    crate::execution::orders::QtySpec::Units(q) => {
                        Some((o.id, o.position_id, q, o.filled_qty))
                    }
                    _ => None,
                })
                .collect();
            let positions =
                k.position_snapshots().iter().map(|p| (p.position_id, p.size)).collect();
            Ok((orders, positions, k.is_hedging()))
        })?;
        let held = |named: Option<u64>| -> Option<f64> {
            match named {
                Some(id) if hedging => positions.iter().find(|p| p.0 == id).map(|p| p.1),
                _ => Some(positions.iter().map(|p| p.1).sum()).filter(|n: &f64| *n > 0.0),
            }
        };
        // Where resting protection counts against the line, the legs of a kind (stops, targets)
        // that rest for one position must not add up to more than it holds: each leg judged
        // alone against the whole position (the loop below) lets a part sale after an add leave
        // legs of 10 and 10 on a position of 15, and a gap through them sells 20. The newest
        // legs shrink first; one left with nothing is cancelled.
        if self.exits_hold_line {
            // (order id, position it names, stop or target, what it has left, whether the venue
            // takes a change now)
            let legs: Vec<(u64, Option<u64>, bool, f64, f64, bool)> =
                sh.with_book(sid, |b, _| {
                    Ok(b.kernel(local)
                        .open_orders()
                        .iter()
                        .filter(|o| {
                            o.client_id.starts_with("prot-")
                                && o.status != OrderStatus::PendingCancel
                        })
                        .filter_map(|o| {
                            let crate::execution::orders::QtySpec::Units(q) = o.qty else {
                                return None;
                            };
                            let stop = match o.kind {
                                crate::execution::orders::OrderKind::StopMarket { .. } => true,
                                crate::execution::orders::OrderKind::Limit { .. } => false,
                                _ => return None,
                            };
                            let changeable = !matches!(
                                o.status,
                                OrderStatus::Submitted | OrderStatus::PendingReplace
                            );
                            Some((
                                o.id,
                                o.position_id,
                                stop,
                                (q - o.filled_qty).max(0.0),
                                o.filled_qty,
                                changeable,
                            ))
                        })
                        .collect())
                })?;
            let mut groups: Vec<(bool, Option<u64>)> = Vec::new();
            for l in &legs {
                let key = (l.2, if hedging { l.1 } else { None });
                if !groups.contains(&key) {
                    groups.push(key);
                }
            }
            for (stop, named) in groups {
                let Some(size) = held(named) else { continue };
                let mut group: Vec<_> =
                    legs.iter().filter(|l| l.2 == stop && (!hedging || l.1 == named)).collect();
                let mut excess = group.iter().map(|l| l.3).sum::<f64>() - size;
                group.sort_by_key(|l| std::cmp::Reverse(l.0));
                for (order_id, _, _, left, filled, changeable) in group.into_iter().copied() {
                    if excess <= 1e-9 {
                        break;
                    }
                    if !changeable {
                        continue;
                    }
                    let cut = left.min(excess);
                    excess -= cut;
                    if left - cut <= 1e-9 {
                        self.cancel(sh, sid, local, idx, order_id)?;
                    } else {
                        self.modify(
                            sh,
                            sid,
                            local,
                            idx,
                            order_id,
                            Some(left - cut + filled),
                            None,
                            None,
                            None,
                        )?;
                    }
                }
            }
        }
        for (order_id, named, qty, filled) in orders {
            if let Some(size) = held(named) {
                // An order's quantity is its total, what it has filled included: it has `qty -
                // filled` left to sell, and to leave `size` the total is `size + filled`.
                let left = qty - filled;
                // Where resting protection counts against the line, an add is covered by orders
                // of its own (`extend_protection`); a leg is never grown to the whole position.
                // A leg bigger than the position is cut above, with the legs beside it.
                if (left - size).abs() > 1e-9 && !self.exits_hold_line {
                    self.modify(
                        sh,
                        sid,
                        local,
                        idx,
                        order_id,
                        Some(size + filled),
                        None,
                        None,
                        None,
                    )?;
                }
            }
        }
        // Where resting protection counts against the line, a venue leg holds part of the
        // position and the engine's watcher the rest: its quantity is the bare part, or its
        // close would be larger than the part it covers. (position it names, stop or target,
        // what it has left to close), for every leg the venue still holds.
        let legs: Vec<(Option<u64>, bool, f64)> = if self.exits_hold_line {
            sh.with_book(sid, |b, _| {
                Ok(b.kernel(local)
                    .open_orders()
                    .iter()
                    .filter(|o| {
                        o.client_id.starts_with("prot-") && o.status != OrderStatus::PendingCancel
                    })
                    .filter_map(|o| {
                        let crate::execution::orders::QtySpec::Units(q) = o.qty else {
                            return None;
                        };
                        match o.kind {
                            crate::execution::orders::OrderKind::StopMarket { .. } => {
                                Some((o.position_id, true, (q - o.filled_qty).max(0.0)))
                            }
                            crate::execution::orders::OrderKind::Limit { .. } => {
                                Some((o.position_id, false, (q - o.filled_qty).max(0.0)))
                            }
                            _ => None,
                        }
                    })
                    .collect())
            })?
        } else {
            Vec::new()
        };
        let covered = |named: Option<u64>, stop: bool| -> f64 {
            legs.iter()
                .filter(|l| l.1 == stop && (!hedging || l.0.is_none() || l.0 == named))
                .map(|l| l.2)
                .sum()
        };
        for w in self.ledger.borrow_mut().watchers.iter_mut() {
            if w.sid == sid.index() && w.local == local {
                if let Some(size) = held(w.position_id) {
                    w.qty = if self.exits_hold_line {
                        // A watcher of both levels covers what is bare of either.
                        let bare_of = |stop: bool| (size - covered(w.position_id, stop)).max(0.0);
                        match (w.stop.is_some(), w.target.is_some()) {
                            (true, true) => bare_of(true).max(bare_of(false)),
                            (true, false) => bare_of(true),
                            _ => bare_of(false),
                        }
                    } else {
                        size
                    };
                }
            }
        }
        if self.exits_hold_line {
            // A watcher with nothing bare left is covered whole by the venue's legs.
            self.ledger
                .borrow_mut()
                .watchers
                .retain(|w| !(w.sid == sid.index() && w.local == local && w.qty <= 1e-9));
        }
        Ok(())
    }

    /// Move the stop (or target) a position already has to `level`. `false` when it has none,
    /// held or watched.
    ///
    /// Where resting protection counts against the line, a position's cover is several legs (an
    /// add gets legs of its own, `extend_protection`) and the level moves on every one of them,
    /// each keeping its size. Elsewhere the one leg is also resized to `size`, the whole position.
    /// A leg the venue has not accepted cannot be changed: it is cancelled and sent again at the
    /// new level for the same quantity.
    #[allow(clippy::too_many_arguments)]
    fn amend_level(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        position_id: Option<u64>,
        stop: bool,
        level: f64,
        size: f64,
        idx: usize,
        ts: i64,
    ) -> PyResult<bool> {
        // (order id, what it has left to close, whether the venue has accepted it)
        let mut legs: Vec<(u64, f64, bool)> = sh.with_book(sid, |b, _| {
            Ok(b.kernel(local)
                .open_orders()
                .iter()
                .filter(|o| {
                    o.client_id.starts_with("prot-")
                        && !matches!(
                            o.status,
                            OrderStatus::PendingCancel | OrderStatus::PendingReplace
                        )
                        && position_id.is_none_or(|p| o.position_id.is_none_or(|q| q == p))
                        && match o.kind {
                            crate::execution::orders::OrderKind::StopMarket { .. } => stop,
                            crate::execution::orders::OrderKind::Limit { .. } => !stop,
                            _ => false,
                        }
                })
                .map(|o| {
                    let left = match o.qty {
                        crate::execution::orders::QtySpec::Units(q) => (q - o.filled_qty).max(0.0),
                        _ => 0.0,
                    };
                    (o.id, left, o.status != OrderStatus::Submitted)
                })
                .collect())
        })?;
        if !self.exits_hold_line {
            legs.truncate(1);
        }
        let mut moved = !legs.is_empty();
        let (limit, trigger) = if stop { (None, Some(level)) } else { (Some(level), None) };
        let exit = if sh.with_book(sid, |b, _| {
            Ok(b.kernel(local).position_snapshots().first().map(|p| p.direction))
        })? == Some(crate::core::types::Direction::Short)
        {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };
        let mut replaced: Vec<(u64, f64)> = Vec::new();
        for (order_id, left, accepted) in legs {
            let held = if self.exits_hold_line { left } else { size };
            if accepted {
                let units = if self.exits_hold_line { None } else { Some(size) };
                self.modify(sh, sid, local, idx, order_id, units, None, limit, trigger)?;
            } else {
                self.cancel(sh, sid, local, idx, order_id)?;
                if let Some(id) =
                    self.place_leg(sh, sid, local, idx, ts, exit, stop, held, level, position_id)?
                {
                    replaced.push((id, held));
                }
            }
        }
        // The leg that was cancelled was one of a stop and target that cancel each other: the new
        // one is linked to the other half when that is the one leg of its size on the other side.
        for (id, qty) in replaced {
            let partners: Vec<u64> = sh.with_book(sid, |b, _| {
                Ok(b.kernel(local)
                    .open_orders()
                    .iter()
                    .filter(|o| {
                        o.client_id.starts_with("prot-")
                            && o.id != id
                            && o.status != OrderStatus::PendingCancel
                            && position_id.is_none_or(|p| o.position_id.is_none_or(|q| q == p))
                            && match o.kind {
                                crate::execution::orders::OrderKind::StopMarket { .. } => !stop,
                                crate::execution::orders::OrderKind::Limit { .. } => stop,
                                _ => false,
                            }
                            && matches!(o.qty, crate::execution::orders::QtySpec::Units(q)
                                if ((q - o.filled_qty) - qty).abs() < 1e-9)
                    })
                    .map(|o| o.id)
                    .collect())
            })?;
            if let [partner] = partners[..] {
                self.link_oco(sid, local, &[id, partner])?;
            }
        }
        let mut ledger = self.ledger.borrow_mut();
        for w in ledger.watchers.iter_mut().filter(|w| {
            w.sid == sid.index()
                && w.local == local
                && w.position_id == position_id
                && (if stop { w.stop.is_some() } else { w.target.is_some() })
        }) {
            if stop {
                w.stop = Some(level);
                w.trail = None;
            } else {
                w.target = Some(level);
            }
            if !self.exits_hold_line {
                w.qty = size;
            }
            moved = true;
        }
        Ok(moved)
    }

    /// The trail a watched stop carries when the request asks for a trailing one.
    fn trail_of(r: &crate::portfolio::kernel::ProtectRequest) -> Option<Trail> {
        let trail = r.trail.filter(|_| r.entry_price > 0.0)?;
        Some(Trail { fraction: trail / r.entry_price, entry: r.entry_price, high: r.entry_price })
    }

    /// Place the stop and target a new position would have had in-process.
    fn protect(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        r: &crate::portfolio::kernel::ProtectRequest,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        let exit = if r.direction == crate::core::types::Direction::Long {
            OrderSide::Sell
        } else {
            OrderSide::Buy
        };
        let mut legs: Vec<(Option<u64>, &str)> = Vec::new();
        let hedging = sh.with_book(sid, |b, _| Ok(b.kernel(local).is_hedging()))?;
        let position_id = if hedging { Some(r.position_id) } else { None };
        let mut watch = Watch {
            sid: sid.index(),
            local,
            exit,
            qty: r.size,
            stop: None,
            target: None,
            position_id,
            trail: None,
        };
        for (level, kind) in [(r.stop, "stop"), (r.target, "target")] {
            let Some(level) = level else { continue };
            // An add that names a level moves the one the position already has, held by the
            // venue or watched by the engine; it places one only where there is none.
            if r.amend
                && self.amend_level(
                    sh,
                    sid,
                    local,
                    position_id,
                    kind == "stop",
                    level,
                    r.size,
                    idx,
                    ts,
                )?
            {
                continue;
            }
            // The level of a hedge leg whose close would leave a sold leg bare is the engine's
            // whatever the venue could hold: it closes the sold legs left bare first, which a
            // resting order at the venue could not. Any other level is an ordinary resting order
            // (`reconcile_hedge_levels` moves the levels as the group changes).
            let engine_holds = sh.with_book(sid, |b, _| {
                Ok(b.kernel(local).engine_holds_levels(r.direction)
                    && !b.sold_legs_left_bare(local, r.position_id).is_empty())
            })?;
            if self.protective_local || engine_holds {
                if kind == "stop" {
                    watch.stop = Some(level);
                    watch.trail = Self::trail_of(r);
                } else {
                    watch.target = Some(level)
                }
                continue;
            }
            let mut spec;
            if kind == "stop" {
                match r.trail {
                    Some(trail) => {
                        spec = Self::spec(exit, "trailing_stop_market", Some(r.size), true);
                        // A backtest's trail stands a fraction of the highest price below it, so
                        // the order trails basis points of its watermark, not the fixed price
                        // distance the entry price would give.
                        if r.entry_price > 0.0 {
                            spec.offset = Some(trail / r.entry_price * 10_000.0);
                            spec.offset_kind = "bps".to_string();
                        } else {
                            spec.offset = Some(trail);
                        }
                    }
                    None => {
                        spec = Self::spec(exit, "stop_market", Some(r.size), true);
                        spec.trigger_price = Some(level);
                    }
                }
            } else {
                spec = Self::spec(exit, "limit", Some(r.size), true);
                spec.limit_price = Some(level);
            }
            spec.position_id = position_id;
            let n = self.next_prot(sid);
            self.unsupported_sent.set(false);
            let id = self.submit(sh, sid, local, &spec, idx, ts, &format!("prot-{n}"), None)?;
            if self.unsupported_sent.replace(false) {
                // The venue cannot hold this leg: the engine watches it, a trailing stop as a
                // trailing one.
                if kind == "stop" {
                    watch.stop = Some(level);
                    watch.trail = Self::trail_of(r);
                } else {
                    watch.target = Some(level)
                }
            } else {
                legs.push((Some(id), kind));
            }
        }
        if watch.stop.is_some() || watch.target.is_some() {
            self.ledger.borrow_mut().watchers.push(watch);
        }
        if legs.len() == 2 {
            let ids: Vec<u64> = legs.iter().filter_map(|(id, _)| *id).collect();
            self.link_oco(sid, local, &ids)?;
        }
        Ok(())
    }

    /// Cancel what is working to add exposure.
    fn cancel_adding(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        idx: usize,
    ) -> PyResult<()> {
        let ids: Vec<u64> = sh.with_book(sid, |b, _| {
            Ok(b.kernel(local)
                .open_orders()
                .into_iter()
                .filter(|o| {
                    o.is_venue_driven() && !o.reduce_only && o.status != OrderStatus::PendingCancel
                })
                .map(|o| o.id)
                .collect())
        })?;
        for id in ids {
            self.cancel(sh, sid, local, idx, id)?;
        }
        Ok(())
    }

    /// A halt: working orders that would add exposure are cancelled, new
    /// entries are refused, and with `flatten` the positions are closed by
    /// orders.
    pub(super) fn halt(
        &self,
        sh: &Shared,
        sid: StrategyId,
        reason: &str,
        flatten: bool,
    ) -> PyResult<()> {
        // A simulated venue keeps no orders of its own to close with: the strategy's book flattens
        // at its last marks, as it does without a venue, and the venue's copy does the same. A
        // real venue is sent the closes as orders.
        let book_flattens = flatten && self.simulated;
        sh.with_book_mut(sid, |b, _| {
            b.halt(reason, book_flattens);
            Ok(())
        })?;
        if self.simulated {
            if let Some(venue) = self.adapter.borrow_mut().as_simulation() {
                venue.halt(sid.index(), reason, flatten);
            }
        }
        let ts = match &*sh.inner.borrow() {
            Inner::Running(c) => c.current().map_or(0, |cur| cur.event.ts),
            _ => 0,
        };
        let mut idx_of = Vec::new();
        for local in 0..sh.locals(sid) {
            let global = sh.global_of(sid, local)?;
            let idx = with_feed(sh, |feed| Ok(feed.last_ordinal(global)))?;
            self.cancel_adding(sh, sid, local, idx)?;
            idx_of.push(idx);
        }
        if flatten && !book_flattens {
            self.flatten(
                sh,
                sid,
                &idx_of.iter().copied().enumerate().collect::<Vec<_>>(),
                ts,
                false,
            )?;
        }
        Ok(())
    }

    /// Whether a reduce-only market order that closes the position (a `close-` or a `prot-`
    /// answer) is working on it: sent and not ended.
    fn close_working(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        position_id: u64,
    ) -> PyResult<bool> {
        sh.with_book(sid, |b, _| {
            Ok(b.kernel(local).open_orders().iter().any(|o| {
                o.is_venue_driven()
                    && o.reduce_only
                    && matches!(o.kind, crate::execution::orders::OrderKind::Market)
                    && (o.client_id.starts_with("close-") || o.client_id.starts_with("prot-"))
                    && o.position_id.is_none_or(|p| p == position_id)
            }))
        })
    }

    /// Close every position of the strategy with orders. A hedge leg whose close would leave a
    /// sold leg of its margin group without cover goes through [`VenueRoute::cover_hedge`]: the
    /// sold legs are closed first and the hedge once they are, so a flatten never sends a
    /// broker the hedge's close ahead of the legs it covers. `targets` are the (local
    /// instrument, event ordinal) pairs to close; `follow_up` is passed to the covers' closes.
    fn flatten(
        &self,
        sh: &Shared,
        sid: StrategyId,
        targets: &[(usize, usize)],
        ts: i64,
        follow_up: bool,
    ) -> PyResult<()> {
        let mut held = Vec::new();
        for &(local, idx) in targets {
            let ids: Vec<u64> = sh.with_book(sid, |b, _| {
                Ok(b.kernel(local).position_snapshots().iter().map(|p| p.position_id).collect())
            })?;
            held.extend(ids.into_iter().map(|id| (local, idx, id)));
        }
        // The hedges first: covering one sends the close of the sold legs it protects.
        let mut covered = std::collections::HashSet::new();
        for &(local, idx, id) in &held {
            if let Some(position) = self.hedge_to_cover(sh, sid, local, Some(id))? {
                self.cover_hedge(sh, sid, local, position, idx, ts, follow_up)?;
                covered.insert((local, id));
            }
        }
        for (local, idx, id) in held {
            // A hedge waits on its cover; a leg that already has a close working (sent by a cover,
            // or by the flatten of another instrument's square-off) is not closed twice. A close
            // the venue refused is no longer working: the next flatten sends it again.
            let key = (sid.index(), local, id);
            let waiting = covered.contains(&(local, id))
                || self
                    .ledger
                    .borrow()
                    .covers
                    .iter()
                    .any(|c| (c.sid, c.local, c.position_id) == key);
            if waiting || self.close_working(sh, sid, local, id)? {
                continue;
            }
            self.close(sh, sid, local, idx, ts, id, false)?;
        }
        Ok(())
    }

    /// The venue id and strategy of an order, for `push_report`.
    /// Every order sent, in the order of their ids: the venue id, the
    /// strategy, the local instrument, the engine's order id and the request.
    pub(super) fn sent(&self) -> Vec<(String, usize, usize, u64, OrderRequest)> {
        let ledger = self.ledger.borrow();
        let mut out: Vec<_> = ledger
            .by_venue
            .iter()
            .map(|(id, p)| (id.clone(), p.sid, p.local, p.order_id, p.request.clone()))
            .collect();
        // By strategy, then by the order's own count: an id is `s{strategy}-{seq}`.
        out.sort_by_key(|o| {
            (o.1, crate::execution::adapter::parse_client_order_id(&o.0).map(|p| p.1))
        });
        out
    }

    /// `sent`, with whether the venue made the order itself (a registered exit order).
    pub(super) fn sent_with_origin(&self) -> Vec<(String, usize, usize, u64, OrderRequest, bool)> {
        let ledger = self.ledger.borrow();
        let mut out: Vec<_> = ledger
            .by_venue
            .iter()
            .map(|(id, p)| (id.clone(), p.sid, p.local, p.order_id, p.request.clone(), p.external))
            .collect();
        out.sort_by_key(|o| {
            (o.1, crate::execution::adapter::parse_client_order_id(&o.0).map(|p| p.1))
        });
        out
    }

    /// Check the node's execution epoch with the venue.
    pub(super) fn heartbeat(&self) -> PyResult<()> {
        let epoch = self.adapter.borrow_mut().fence(0);
        match epoch {
            Ok(()) => Ok(()),
            Err(e @ (AdapterError::Fenced(_) | AdapterError::Journal(_))) => Err(self.stopped(e)),
            Err(AdapterError::Unavailable(why)) => Err(self.unavailable(why)),
            Err(other) => Err(Fail::new("venue_unavailable", other.to_string()).into()),
        }
    }

    /// Send an order again, as the request the engine first made of it. The
    /// caller has asked the venue whether it holds the order and was told it
    /// does not.
    pub(super) fn resend(&self, venue_id: &str) -> PyResult<()> {
        let Some(placed) = self.placed(venue_id) else {
            return Err(Fail::new(
                "unknown_client_order_id",
                format!("no order was sent under {venue_id:?}"),
            )
            .param("client_order_id")
            .value(venue_id)
            .into());
        };
        if placed.external {
            return Err(Fail::new(
                "unsupported_in_venue_run",
                format!("{venue_id:?} is an order the venue makes itself (an exit rule it holds): it cannot be resent"),
            )
            .param("client_order_id")
            .value(venue_id)
            .into());
        }
        let sent = self.adapter.borrow_mut().submit(&placed.request);
        match sent {
            Ok(()) => Ok(()),
            Err(AdapterError::Refused(why) | AdapterError::Unsupported(why)) => {
                Err(Fail::new("venue_unavailable", format!("the venue refused the resend: {why}"))
                    .into())
            }
            Err(AdapterError::Unavailable(why)) => Err(self.unavailable(why)),
            Err(e @ (AdapterError::Fenced(_) | AdapterError::Journal(_))) => Err(self.stopped(e)),
        }
    }

    /// Register an order the venue makes on its own (an exit rule it holds
    /// fires `{rule}:x{n}`), so that its reports are booked like any order's.
    /// The engine did not send it and never resends it. The same id again is a
    /// no-op.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn register_external(
        &self,
        sh: &Shared,
        sid: StrategyId,
        local: usize,
        venue_id: &str,
        side: OrderSide,
        quantity: f64,
        idx: usize,
        ts: i64,
    ) -> PyResult<()> {
        if self.placed(venue_id).is_some() {
            return Ok(());
        }
        let global = sh.global_of(sid, local)?;
        let symbol = sh.store.symbols().symbol(global).unwrap_or_default().to_string();
        let (order_id, sub_idx) = sh.with_book_mut(sid, |book, current| {
            let sub_idx = book.submission_idx(local, idx, current);
            let id = book.kernel_mut(local).submit_venue_order(
                side,
                quantity,
                crate::execution::orders::OrderKind::Market,
                crate::execution::orders::TimeInForce::Day,
                sub_idx,
                ts,
                venue_id.to_string(),
                false,
                true,
                None,
                None,
                None,
            );
            Ok((id, sub_idx))
        })?;
        let mut tags = std::collections::BTreeMap::new();
        tags.insert("client_id".to_string(), venue_id.to_string());
        tags.insert("held_by".to_string(), "venue".to_string());
        let request = OrderRequest {
            client_order_id: venue_id.to_string(),
            symbol,
            side,
            quantity,
            kind: crate::execution::orders::OrderKind::Market,
            tif: crate::execution::orders::TimeInForce::Day,
            strategy_id: sid.index() as u32,
            account_id: self.account_id.clone(),
            reduce_only: true,
            post_only: false,
            parent_client_order_id: None,
            tags: tags.into_iter().collect(),
            submitted_ns: ts,
            submitted_idx: sub_idx,
            size_frac: None,
            position_id: None,
        };
        let mut ledger = self.ledger.borrow_mut();
        ledger.by_client.insert((sid.index(), venue_id.to_string()), venue_id.to_string());
        ledger.by_engine.insert((sid.index(), local, order_id), venue_id.to_string());
        ledger.by_venue.insert(
            venue_id.to_string(),
            Placed {
                sid: sid.index(),
                local,
                order_id,
                request,
                unsent: false,
                ended: false,
                external: true,
            },
        );
        Ok(())
    }

    /// End registered exit orders (the venue's rule was cancelled or replaced): each is
    /// cancelled in the book as the venue would have ended it, so none stays working for a
    /// rule the venue no longer holds. An order that already ended is left as it is; an id
    /// that is not a registered exit order is an error.
    pub(super) fn end_external(
        &self,
        sh: &Shared,
        sid: StrategyId,
        ids: &[String],
        ts: i64,
    ) -> PyResult<()> {
        for id in ids {
            let Some(placed) = self.placed(id) else {
                if self.is_dropped(id) {
                    continue;
                }
                return Err(Fail::new(
                    "unknown_client_order_id",
                    format!("{id:?} is not an order this node registered"),
                )
                .param("client_order_id")
                .value(id.as_str())
                .into());
            };
            if !placed.external || placed.sid != sid.index() {
                return Err(Fail::new(
                    "invalid_execution",
                    format!("{id:?} is not an exit order the venue holds for this strategy"),
                )
                .param("client_order_id")
                .value(id.as_str())
                .into());
            }
            let global = sh.global_of(sid, placed.local)?;
            let idx = with_feed(sh, |feed| Ok(feed.last_ordinal(global)))?;
            self.book_silently(
                sh,
                sid,
                placed.local,
                placed.order_id,
                idx,
                ts,
                &ExecutionReport::CancelAck { ts_ns: ts },
            )?;
        }
        Ok(())
    }

    /// Ids of the orders the venue makes itself that are still working: an exit rule it
    /// holds has not fired them yet, so the venue's order book does not list them.
    pub(super) fn held_ids(&self, sh: &Shared) -> PyResult<Vec<String>> {
        let held: Vec<(String, usize, u64, usize)> = {
            let ledger = self.ledger.borrow();
            ledger
                .by_venue
                .iter()
                .filter(|(_, p)| p.external)
                .map(|(id, p)| (id.clone(), p.sid, p.order_id, p.local))
                .collect()
        };
        let mut out = Vec::new();
        for (id, sid, order_id, local) in held {
            let working = sh.with_book(StrategyId::new(sid as u32), |b, _| {
                Ok(b.kernel(local).order(order_id).is_some_and(|o| !o.status.is_terminal()))
            })?;
            if working {
                out.push(id);
            }
        }
        out.sort();
        Ok(out)
    }

    pub(super) fn lookup(&self, venue_id: &str) -> Option<(usize, usize)> {
        self.placed(venue_id).map(|p| (p.sid, p.local))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOSE: i64 = 1_000;
    const NEXT_OPEN: i64 = 5_000;

    #[test]
    fn an_expired_leg_lapsed_unless_it_was_being_cancelled() {
        let lapsed = |before| {
            lapsed_by_session(before, OrderStatus::Expired, CLOSE, Some(CLOSE), Some(NEXT_OPEN))
        };
        assert!(lapsed(OrderStatus::Accepted));
        assert!(lapsed(OrderStatus::PartiallyFilled));
        // the engine or the strategy asked for it to go: it was taken off, not lapsed
        assert!(!lapsed(OrderStatus::PendingCancel));
        // a repeated report of an order already over changes nothing
        assert!(!lapsed(OrderStatus::Expired));
    }

    #[test]
    fn a_cancel_nobody_asked_for_lapsed_only_between_the_close_and_the_next_open() {
        let at = |ts| {
            lapsed_by_session(
                OrderStatus::Accepted,
                OrderStatus::Canceled,
                ts,
                Some(CLOSE),
                Some(NEXT_OPEN),
            )
        };
        assert!(!at(CLOSE - 1), "inside the session: the venue's own cancel, not a lapse");
        assert!(at(CLOSE));
        assert!(at(NEXT_OPEN));
        assert!(!at(NEXT_OPEN + 1), "in the next session: not the close's doing");
        // asked for, it is a cancel whatever the time
        assert!(!lapsed_by_session(
            OrderStatus::PendingCancel,
            OrderStatus::Canceled,
            CLOSE,
            Some(CLOSE),
            Some(NEXT_OPEN)
        ));
        // without a session there is no close to lapse at: a cancel is a cancel
        assert!(!lapsed_by_session(
            OrderStatus::Accepted,
            OrderStatus::Canceled,
            CLOSE,
            None,
            None
        ));
        // a fill or a rejection is not a lapse
        for after in [OrderStatus::Filled, OrderStatus::Rejected] {
            assert!(!lapsed_by_session(OrderStatus::Accepted, after, CLOSE, Some(CLOSE), None));
        }
    }

    #[test]
    fn a_lapsed_leg_is_due_at_the_first_event_in_session_after_its_close() {
        assert!(!rearm_due(CLOSE, CLOSE, true), "the close itself is the old session");
        assert!(!rearm_due(CLOSE, CLOSE + 1, false), "after the close, before the open");
        assert!(rearm_due(CLOSE, NEXT_OPEN + 1, true));
    }

    #[test]
    fn a_bar_crosses_a_level_on_the_side_the_level_closes_from() {
        // a long's stop at 90 (exit a sell): a bar with low 84 is through it, one with low 91 not
        assert!(leg_crossed(true, OrderSide::Sell, 90.0, 86.0, 84.0));
        assert!(!leg_crossed(true, OrderSide::Sell, 90.0, 95.0, 91.0));
        // a long's target at 110: through it at a high of 112, not at 109
        assert!(leg_crossed(false, OrderSide::Sell, 110.0, 112.0, 100.0));
        assert!(!leg_crossed(false, OrderSide::Sell, 110.0, 109.0, 100.0));
        // a short's stop at 110 (exit a buy): through at a high of 110; its target at 90 at a low
        // of 90
        assert!(leg_crossed(true, OrderSide::Buy, 110.0, 110.0, 100.0));
        assert!(leg_crossed(false, OrderSide::Buy, 90.0, 100.0, 90.0));
        assert!(!leg_crossed(false, OrderSide::Buy, 90.0, 100.0, 91.0));
    }

    #[test]
    fn a_trailing_legs_level_stands_its_offset_from_the_watermark() {
        use crate::execution::orders::TrailOffset;
        // 500 bps of a watermark of 120 is 6: a sell stop at 120 - 6 = 114, a buy stop at 126
        let leg = LapsedLeg::Trail { offset: TrailOffset::Bps(500.0), watermark: Some(120.0) };
        assert_eq!(leg.level(OrderSide::Sell), Some(114.0));
        assert_eq!(leg.level(OrderSide::Buy), Some(126.0));
        let none = LapsedLeg::Trail { offset: TrailOffset::Price(2.0), watermark: None };
        assert_eq!(none.level(OrderSide::Sell), None);
        assert_eq!(LapsedLeg::Stop(90.0).level(OrderSide::Sell), Some(90.0));
        assert!(LapsedLeg::Stop(90.0).is_stop() && !LapsedLeg::Target(110.0).is_stop());
    }
}
