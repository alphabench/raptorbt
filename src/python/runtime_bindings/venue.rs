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
    /// Sold legs a cover has sent a close for: `(strategy, local instrument, position id)`.
    cover_sent: std::collections::HashSet<(usize, usize, u64)>,
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
}

/// The venue a node sends its orders to, and what it has sent.
pub(super) struct VenueRoute {
    adapter: RefCell<Box<dyn ExecutionAdapter>>,
    /// The adapter simulates an exchange that advances with the market.
    simulated: bool,
    account_id: Option<String>,
    /// An exception a Python adapter raised that is not a refusal, kept to
    /// be re-raised as it was.
    raised: Rc<RefCell<Option<PyErr>>>,
    ledger: RefCell<Ledger>,
    /// Protective orders are watched by the engine (`protective="local"`).
    protective_local: bool,
    /// The adapter said it cannot take the last order sent.
    unsupported_sent: std::cell::Cell<bool>,
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
        Self {
            adapter: RefCell::new(adapter),
            simulated,
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

    /// [`Self::note_ended`] when the order has reached a final state.
    fn note_if_ended(&self, sh: &Shared, sid: StrategyId, local: usize, order_id: u64) {
        let ended = sh
            .with_book(sid, |b, _| {
                Ok(b.kernel(local).order(order_id).is_some_and(|o| o.status.is_terminal()))
            })
            .unwrap_or(false);
        if ended {
            if let Some(venue_id) = self.venue_id_of(sid, local, order_id) {
                self.note_ended(&venue_id);
            }
        }
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
        // An order that names no time in force takes its instrument's default.
        let resolved;
        let spec = if spec.tif == super::super::strategy_bindings::TIF_AUTO {
            let tif = sh.with_book(sid, |b, _| Ok(b.kernel(local).default_tif()))?;
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
            let verdict = sh.with_book_mut(sid, |book, _| {
                Ok(book.venue_pretrade(local, parsed.side, units, spec.reduce_only, price))
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
        let mut tags = spec.tags.clone();
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
            Err(AdapterError::Unavailable(why)) => Err(self.unavailable(why)),
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
        sh.with_book_mut(sid, |book, _| {
            let (_, events) =
                book.apply_venue_report(local, idx, ts, order_id, report).map_err(venue_err)?;
            book.kernel_mut(local).pending_events.extend(events);
            Ok(())
        })?;
        self.note_if_ended(sh, sid, local, order_id);
        Ok(())
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
        let change = OrderChange { quantity: units, limit_price, trigger_price };
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
            tif: "gtc".to_string(),
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
        self.note_if_ended(sh, st.sid, placed.local, placed.order_id);
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
            tif: "gtc".to_string(),
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
            let ids: Vec<u64> = sh.with_book(sid, |b, _| {
                Ok(b.kernel(local).position_snapshots().iter().map(|p| p.position_id).collect())
            })?;
            for id in ids {
                // A hedge leg's square-off closes the sold legs of its group first.
                match self.hedge_to_cover(sh, sid, local, Some(id))? {
                    Some(position) => self.cover_hedge(sh, sid, local, position, idx, ts, true)?,
                    None => self.close(sh, sid, local, idx, ts, id, false)?,
                }
            }
        }
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
            ledger.watchers.retain(|w| {
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
            if !self.ledger.borrow_mut().cover_sent.insert((sid.index(), j, id)) {
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
        // A sold leg that has closed no longer waits on its close.
        let live: Vec<(usize, usize, u64)> = {
            let sent: Vec<_> = self.ledger.borrow().cover_sent.iter().copied().collect();
            let mut live = Vec::new();
            for (s, l, p) in sent {
                if s != sid.index() {
                    continue;
                }
                let held = sh.with_book(sid, |b, _| {
                    Ok(b.kernel(l).position_snapshots().iter().any(|x| x.position_id == p))
                })?;
                if held {
                    live.push((s, l, p));
                }
            }
            live
        };
        self.ledger.borrow_mut().cover_sent.retain(|k| k.0 != sid.index() || live.contains(k));
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
                self.take_levels(sh, sid, m, position_id, size, named, idx)?;
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
                        entry_price: 0.0,
                        stop: w.stop,
                        target: w.target,
                        trail: None,
                        position_id,
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
        size: f64,
        named: Option<u64>,
        idx: usize,
    ) -> PyResult<()> {
        let orders: Vec<(u64, Option<u64>, crate::execution::orders::OrderKind)> =
            sh.with_book(sid, |b, _| {
                Ok(b.kernel(local)
                    .open_orders()
                    .iter()
                    .filter(|o| {
                        o.client_id.starts_with("prot-")
                            && o.reduce_only
                            && o.status != OrderStatus::PendingCancel
                    })
                    .map(|o| (o.id, o.position_id, o.kind))
                    .collect())
            })?;
        let mut watch = Watch {
            sid: sid.index(),
            local,
            exit: OrderSide::Sell,
            qty: size,
            stop: None,
            target: None,
            position_id: named,
        };
        let mut taken = Vec::new();
        for (order_id, order_position, kind) in orders {
            if order_position.is_some_and(|p| p != position_id) {
                continue;
            }
            match kind {
                crate::execution::orders::OrderKind::StopMarket { trigger } => {
                    watch.stop = Some(trigger)
                }
                crate::execution::orders::OrderKind::Limit { price } => watch.target = Some(price),
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
        };
        for (level, kind) in [(r.stop, "stop"), (r.target, "target")] {
            let Some(level) = level else { continue };
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
                    watch.stop = Some(level)
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
                        spec.offset = Some(trail);
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
                // The venue cannot hold this leg: the engine watches it.
                if kind == "stop" {
                    watch.stop = Some(level)
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
        for local in 0..sh.locals(sid) {
            let global = sh.global_of(sid, local)?;
            let idx = with_feed(sh, |feed| Ok(feed.last_ordinal(global)))?;
            self.cancel_adding(sh, sid, local, idx)?;
            if flatten && !book_flattens {
                let ids: Vec<u64> = sh.with_book(sid, |b, _| {
                    Ok(b.kernel(local).position_snapshots().iter().map(|p| p.position_id).collect())
                })?;
                for id in ids {
                    self.close(sh, sid, local, idx, ts, id, false)?;
                }
            }
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
