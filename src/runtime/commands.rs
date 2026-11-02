//! A strategy's queued work, as typed data.
//!
//! Hooks queue two kinds of work while they run: *commands* (typed orders,
//! cancels, closes, ...) and *intents* (the legacy `enter` / `close_position`
//! requests, which only a step can carry). Both are plain data here, with no
//! interpreter objects in them: an order is converted to an [`OrderSpec`] once,
//! when it is submitted.
//!
//! [`CommandState`] then routes that work to an engine through a [`Target`]:
//! the client-id map, the schedule map, parent and one-cancels-other checks,
//! and the intents held for an instrument's next step.

use std::collections::HashMap;

/// A typed order, flattened to what the engine's submit takes.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderSpec {
    pub side: String,
    pub kind: String,
    pub units: Option<f64>,
    pub size_frac: Option<f64>,
    pub limit_price: Option<f64>,
    pub trigger_price: Option<f64>,
    pub tif: String,
    pub expire_ns: Option<i64>,
    pub stop_price: Option<f64>,
    pub target_price: Option<f64>,
    pub offset: Option<f64>,
    pub offset_kind: String,
    pub limit_offset: f64,
    pub post_only: bool,
    pub reduce_only: bool,
}

/// An execution schedule: a TWAP releases its own slices.
#[derive(Debug, Clone, PartialEq)]
pub struct TwapSpec {
    pub units: f64,
    pub side: String,
    pub slices: u32,
    pub interval_ns: i64,
    pub reduce_only: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Order {
    Plain(OrderSpec),
    Twap(TwapSpec),
}

/// Work routed to the engine outside a step.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Submit {
        client_id: String,
        order: Box<Order>,
        parent: Option<String>,
        symbol: Option<String>,
    },
    Cancel(String),
    CancelAll,
    LinkOco(Vec<String>),
    Close {
        position_id: u64,
        symbol: Option<String>,
    },
    CloseAllFor(String),
    Modify {
        client_id: String,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    },
}

/// A legacy request that rides a step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Intent {
    Enter { size_frac: Option<f64>, stop_price: Option<f64>, target_price: Option<f64> },
    Close,
}

/// What a strategy has queued and the engine has not yet seen.
#[derive(Debug, Default)]
pub struct Queue {
    commands: Vec<Command>,
    intents: Vec<Intent>,
}

impl Queue {
    pub fn push_command(&mut self, command: Command) {
        self.commands.push(command);
    }

    pub fn push_intent(&mut self, intent: Intent) {
        self.intents.push(intent);
    }

    pub fn has_commands(&self) -> bool {
        !self.commands.is_empty()
    }

    pub fn has_intents(&self) -> bool {
        !self.intents.is_empty()
    }

    pub fn take_commands(&mut self) -> Vec<Command> {
        std::mem::take(&mut self.commands)
    }

    pub fn take_intents(&mut self) -> Vec<Intent> {
        std::mem::take(&mut self.intents)
    }

    /// Put `held` ahead of whatever is queued.
    pub fn restore_intents(&mut self, held: Vec<Intent>) {
        self.intents.splice(0..0, held);
    }
}

// ---------------------------------------------------------------------------
// Routing queued work to the engine
// ---------------------------------------------------------------------------

/// A value named in a refusal.
#[derive(Debug, Clone, PartialEq)]
pub enum ErrVal {
    Str(String),
    List(Vec<String>),
}

/// A refusal made by the router itself, in the shape the structured errors
/// take: a registered code, the argument at fault, what was supplied and the
/// alternatives when they depend on the call.
#[derive(Debug, Clone, PartialEq)]
pub struct CmdError {
    pub code: &'static str,
    pub message: String,
    pub parameter: &'static str,
    pub value: ErrVal,
    /// The accepted alternatives; `Some` of none is an explicit empty list.
    pub valid: Option<Vec<String>>,
}

impl CmdError {
    fn link(message: &str, parameter: &'static str, value: ErrVal) -> Self {
        Self {
            code: "invalid_order_link",
            message: message.to_string(),
            parameter,
            value,
            valid: None,
        }
    }
}

/// The engine a run's commands are routed to: one instrument or many.
///
/// `Error` carries whatever the engine call can fail with; the checks the
/// router makes itself are raised through [`Target::refuse`].
pub trait Target {
    type Error;

    fn refuse(error: CmdError) -> Self::Error;
    fn n_instruments(&self) -> usize;
    /// The instrument a symbol names, or `current` for no symbol.
    fn instrument_of(&mut self, symbol: Option<&str>, current: usize)
        -> Result<usize, Self::Error>;
    fn submit_order(
        &mut self,
        instrument: usize,
        spec: &OrderSpec,
        idx: usize,
        ts: i64,
        client_id: &str,
        parent: Option<u64>,
    ) -> Result<u64, Self::Error>;
    fn submit_twap(
        &mut self,
        instrument: usize,
        spec: &TwapSpec,
        idx: usize,
        ts: i64,
        client_id: &str,
    ) -> Result<u64, Self::Error>;
    fn cancel_twap(
        &mut self,
        instrument: usize,
        algo_id: u64,
        idx: usize,
    ) -> Result<(), Self::Error>;
    fn cancel_order(
        &mut self,
        instrument: usize,
        idx: usize,
        order_id: u64,
    ) -> Result<(), Self::Error>;
    fn cancel_all_orders(&mut self, instrument: usize, idx: usize) -> Result<(), Self::Error>;
    fn link_oco(&mut self, instrument: usize, order_ids: &[u64]) -> Result<(), Self::Error>;
    fn request_close(&mut self, instrument: usize, position_id: u64) -> Result<(), Self::Error>;
    fn position_ids(&mut self, instrument: usize) -> Result<Vec<u64>, Self::Error>;
    fn modify_order(
        &mut self,
        instrument: usize,
        order_id: u64,
        units: Option<f64>,
        size_frac: Option<f64>,
        limit_price: Option<f64>,
        trigger_price: Option<f64>,
    ) -> Result<(), Self::Error>;
}

/// Why folding intents into a step failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldError {
    DuplicateEntry,
    DuplicateClose,
}

impl FoldError {
    /// The message, naming the event (`where_`) it happened on.
    pub fn message(self, where_: &str) -> String {
        match self {
            Self::DuplicateEntry => format!("duplicate entry intents queued on {where_}"),
            Self::DuplicateClose => format!("duplicate close intents queued on {where_}"),
        }
    }

    /// The refusal, naming the event (`where_`) it happened on.
    pub fn refusal(self, where_: &str) -> CmdError {
        CmdError {
            code: "invalid_order_intent",
            message: self.message(where_),
            parameter: "intents",
            value: ErrVal::Str(
                match self {
                    Self::DuplicateEntry => "entry",
                    Self::DuplicateClose => "close",
                }
                .to_string(),
            ),
            valid: None,
        }
    }
}

/// Intents folded into the arguments of one step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Folded {
    pub entry: bool,
    pub exit: bool,
    pub size_mult: Option<f64>,
    pub stop_price: Option<f64>,
    pub target_price: Option<f64>,
}

/// A run's state for routing a strategy's work: where each client order
/// went, and the intents no step could carry.
#[derive(Debug, Default)]
pub struct CommandState {
    /// Client order id -> (instrument, engine id), for the life of the run.
    ids: HashMap<String, (usize, u64)>,
    /// The client order ids in the order first seen, for refusals that list them.
    order: Vec<String>,
    /// Schedules live apart from orders: an algo id is not an order id, so a
    /// modify, parent or OCO link naming one must not resolve to it.
    algos: HashMap<String, (usize, u64)>,
    /// Intents queued where no step could carry them, by instrument key.
    held: HashMap<usize, Vec<Intent>>,
}

impl CommandState {
    /// Whether any instrument has intents waiting for its next step.
    pub fn has_held(&self) -> bool {
        !self.held.is_empty()
    }

    /// Route `commands` to their instruments through `target`. `current` is
    /// the event's instrument, `idx` its ordinal there and `ts` its time.
    pub fn apply<T: Target>(
        &mut self,
        target: &mut T,
        commands: Vec<Command>,
        current: usize,
        idx: usize,
        ts: i64,
    ) -> Result<(), T::Error> {
        for command in commands {
            match command {
                Command::Submit { client_id, order, parent, symbol } => {
                    let instrument = target.instrument_of(symbol.as_deref(), current)?;
                    let spec = match *order {
                        Order::Twap(spec) => {
                            // A schedule, not an order: it releases its own slices.
                            let algo =
                                target.submit_twap(instrument, &spec, idx, ts, &client_id)?;
                            self.algos.insert(client_id, (instrument, algo));
                            continue;
                        }
                        Order::Plain(spec) => spec,
                    };
                    let parent_id = match parent.as_deref() {
                        None | Some("") => None,
                        Some(name) => {
                            let Some(&(owner, engine)) = self.ids.get(name) else {
                                return Err(T::refuse(CmdError {
                                    code: "invalid_order_link",
                                    message: format!("unknown parent order {}", py_repr(name)),
                                    parameter: "parent",
                                    value: ErrVal::Str(name.to_string()),
                                    valid: Some(self.order.clone()),
                                }));
                            };
                            if owner != instrument {
                                return Err(T::refuse(CmdError::link(
                                    "parent order belongs to a different symbol",
                                    "parent",
                                    ErrVal::Str(name.to_string()),
                                )));
                            }
                            Some(engine)
                        }
                    };
                    let engine =
                        target.submit_order(instrument, &spec, idx, ts, &client_id, parent_id)?;
                    if self.ids.insert(client_id.clone(), (instrument, engine)).is_none() {
                        self.order.push(client_id);
                    }
                }
                Command::Cancel(client_id) => {
                    if let Some(&(instrument, id)) = self.ids.get(&client_id) {
                        target.cancel_order(instrument, idx, id)?;
                    } else if let Some(&(instrument, algo)) = self.algos.get(&client_id) {
                        // A finished schedule is gone from the engine; cancelling
                        // it does nothing, as for a filled order.
                        target.cancel_twap(instrument, algo, idx)?;
                    }
                }
                Command::CancelAll => {
                    for instrument in 0..target.n_instruments() {
                        target.cancel_all_orders(instrument, idx)?;
                    }
                }
                Command::LinkOco(client_ids) => {
                    let mapped: Vec<(usize, u64)> =
                        client_ids.iter().filter_map(|c| self.ids.get(c).copied()).collect();
                    if mapped.len() >= 2 {
                        if mapped.iter().any(|m| m.0 != mapped[0].0) {
                            return Err(T::refuse(CmdError::link(
                                "one-cancels-other links cannot span symbols",
                                "client_ids",
                                ErrVal::List(client_ids),
                            )));
                        }
                        let ids: Vec<u64> = mapped.iter().map(|m| m.1).collect();
                        target.link_oco(mapped[0].0, &ids)?;
                    }
                }
                Command::Close { position_id, symbol } => {
                    let instrument = target.instrument_of(symbol.as_deref(), current)?;
                    target.request_close(instrument, position_id)?;
                }
                Command::CloseAllFor(symbol) => {
                    let instrument = target.instrument_of(Some(&symbol), current)?;
                    for position_id in target.position_ids(instrument)? {
                        target.request_close(instrument, position_id)?;
                    }
                }
                Command::Modify { client_id, units, size_frac, limit_price, trigger_price } => {
                    // The id map carries the owning instrument, so a modify
                    // routes without the caller naming a symbol.
                    if let Some(&(instrument, id)) = self.ids.get(&client_id) {
                        target.modify_order(
                            instrument,
                            id,
                            units,
                            size_frac,
                            limit_price,
                            trigger_price,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Fold the intents for the step of instrument `key`: those held for it
    /// by [`CommandState::hold`] ride ahead of anything queued since.
    pub fn fold(&mut self, queue: &mut Queue, key: usize) -> Result<Folded, FoldError> {
        if !self.held.is_empty() {
            if let Some(held) = self.held.remove(&key) {
                queue.restore_intents(held);
            }
        }
        let mut folded = Folded::default();
        for intent in queue.take_intents() {
            match intent {
                Intent::Enter { size_frac, stop_price, target_price } => {
                    if folded.entry {
                        return Err(FoldError::DuplicateEntry);
                    }
                    folded.entry = true;
                    folded.size_mult = size_frac;
                    folded.stop_price = stop_price;
                    folded.target_price = target_price;
                }
                Intent::Close => {
                    if folded.exit {
                        return Err(FoldError::DuplicateClose);
                    }
                    folded.exit = true;
                }
            }
        }
        Ok(folded)
    }

    /// Keep what is queued for the next step of instrument `key`, rather than
    /// leaving it on the queue for whichever instrument is stepped next.
    pub fn hold(&mut self, queue: &mut Queue, key: usize) {
        if queue.has_intents() {
            self.held.entry(key).or_default().extend(queue.take_intents());
        }
    }
}

/// `repr()` of a Python `str`, for messages that quote a client id.
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Mock {
        log: Vec<String>,
        next: u64,
        positions: Vec<u64>,
    }

    impl Target for Mock {
        type Error = String;

        fn refuse(error: CmdError) -> String {
            error.message
        }
        fn n_instruments(&self) -> usize {
            2
        }
        fn instrument_of(&mut self, symbol: Option<&str>, current: usize) -> Result<usize, String> {
            match symbol {
                None | Some("") => Ok(current),
                Some("A") => Ok(0),
                Some("B") => Ok(1),
                Some(other) => Err(format!("no {other}")),
            }
        }
        fn submit_order(
            &mut self,
            instrument: usize,
            _spec: &OrderSpec,
            idx: usize,
            _ts: i64,
            client_id: &str,
            parent: Option<u64>,
        ) -> Result<u64, String> {
            self.next += 1;
            self.log.push(format!("submit {client_id} on {instrument} @{idx} parent={parent:?}"));
            Ok(self.next)
        }
        fn submit_twap(
            &mut self,
            instrument: usize,
            _spec: &TwapSpec,
            _idx: usize,
            _ts: i64,
            client_id: &str,
        ) -> Result<u64, String> {
            self.next += 100;
            self.log.push(format!("twap {client_id} on {instrument}"));
            Ok(self.next)
        }
        fn cancel_twap(
            &mut self,
            instrument: usize,
            algo_id: u64,
            _idx: usize,
        ) -> Result<(), String> {
            self.log.push(format!("cancel_twap {instrument} {algo_id}"));
            Ok(())
        }
        fn cancel_order(
            &mut self,
            instrument: usize,
            _idx: usize,
            order_id: u64,
        ) -> Result<(), String> {
            self.log.push(format!("cancel {instrument} {order_id}"));
            Ok(())
        }
        fn cancel_all_orders(&mut self, instrument: usize, _idx: usize) -> Result<(), String> {
            self.log.push(format!("cancel_all {instrument}"));
            Ok(())
        }
        fn link_oco(&mut self, instrument: usize, order_ids: &[u64]) -> Result<(), String> {
            self.log.push(format!("oco {instrument} {order_ids:?}"));
            Ok(())
        }
        fn request_close(&mut self, instrument: usize, position_id: u64) -> Result<(), String> {
            self.log.push(format!("close {instrument} {position_id}"));
            Ok(())
        }
        fn position_ids(&mut self, _instrument: usize) -> Result<Vec<u64>, String> {
            Ok(self.positions.clone())
        }
        fn modify_order(
            &mut self,
            instrument: usize,
            order_id: u64,
            units: Option<f64>,
            _size_frac: Option<f64>,
            _limit_price: Option<f64>,
            _trigger_price: Option<f64>,
        ) -> Result<(), String> {
            self.log.push(format!("modify {instrument} {order_id} {units:?}"));
            Ok(())
        }
    }

    fn spec() -> OrderSpec {
        OrderSpec {
            side: "buy".into(),
            kind: "market".into(),
            units: Some(1.0),
            size_frac: None,
            limit_price: None,
            trigger_price: None,
            tif: "gtc".into(),
            expire_ns: None,
            stop_price: None,
            target_price: None,
            offset: None,
            offset_kind: "price".into(),
            limit_offset: 0.0,
            post_only: false,
            reduce_only: false,
        }
    }

    fn submit(id: &str, parent: Option<&str>, symbol: Option<&str>) -> Command {
        Command::Submit {
            client_id: id.into(),
            order: Box::new(Order::Plain(spec())),
            parent: parent.map(String::from),
            symbol: symbol.map(String::from),
        }
    }

    #[test]
    fn parent_must_be_known_and_on_the_same_symbol() {
        let mut state = CommandState::default();
        let mut mock = Mock::default();
        let err =
            state.apply(&mut mock, vec![submit("c", Some("nope"), None)], 0, 3, 9).unwrap_err();
        assert_eq!(err, "unknown parent order 'nope'");
        state.apply(&mut mock, vec![submit("p", None, Some("A"))], 0, 3, 9).unwrap();
        let err =
            state.apply(&mut mock, vec![submit("c", Some("p"), Some("B"))], 0, 3, 9).unwrap_err();
        assert_eq!(err, "parent order belongs to a different symbol");
        state.apply(&mut mock, vec![submit("c", Some("p"), Some("A"))], 0, 3, 9).unwrap();
        assert_eq!(mock.log.last().unwrap(), "submit c on 0 @3 parent=Some(1)");
    }

    #[test]
    fn oco_links_ignore_unknown_ids_and_refuse_to_span_symbols() {
        let mut state = CommandState::default();
        let mut mock = Mock::default();
        let cmds = vec![submit("a", None, Some("A")), submit("b", None, Some("A"))];
        state.apply(&mut mock, cmds, 0, 0, 0).unwrap();
        state.apply(&mut mock, vec![submit("x", None, Some("B"))], 0, 0, 0).unwrap();
        let link = |ids: &[&str]| Command::LinkOco(ids.iter().map(|s| s.to_string()).collect());
        state.apply(&mut mock, vec![link(&["a", "ghost"])], 0, 0, 0).unwrap();
        assert!(!mock.log.iter().any(|l| l.starts_with("oco")));
        state.apply(&mut mock, vec![link(&["a", "ghost", "b"])], 0, 0, 0).unwrap();
        assert_eq!(mock.log.last().unwrap(), "oco 0 [1, 2]");
        let err = state.apply(&mut mock, vec![link(&["a", "x"])], 0, 0, 0).unwrap_err();
        assert_eq!(err, "one-cancels-other links cannot span symbols");
    }

    #[test]
    fn cancel_reaches_orders_then_schedules_and_ignores_unknown() {
        let mut state = CommandState::default();
        let mut mock = Mock::default();
        let twap = Command::Submit {
            client_id: "t".into(),
            order: Box::new(Order::Twap(TwapSpec {
                units: 4.0,
                side: "buy".into(),
                slices: 2,
                interval_ns: 5,
                reduce_only: false,
            })),
            parent: None,
            symbol: Some("B".into()),
        };
        state.apply(&mut mock, vec![twap, submit("o", None, None)], 0, 0, 0).unwrap();
        let cancel = |s: &str| Command::Cancel(s.into());
        state.apply(&mut mock, vec![cancel("o"), cancel("t"), cancel("ghost")], 0, 1, 1).unwrap();
        assert_eq!(&mock.log[mock.log.len() - 2..], ["cancel 0 101", "cancel_twap 1 100"]);
        // A schedule id is not an order id: it cannot be a parent or modified.
        let err = state.apply(&mut mock, vec![submit("c", Some("t"), None)], 0, 0, 0).unwrap_err();
        assert_eq!(err, "unknown parent order 't'");
        let before = mock.log.len();
        let modify = Command::Modify {
            client_id: "t".into(),
            units: Some(1.0),
            size_frac: None,
            limit_price: None,
            trigger_price: None,
        };
        state.apply(&mut mock, vec![modify], 0, 0, 0).unwrap();
        assert_eq!(mock.log.len(), before);
    }

    #[test]
    fn close_all_for_and_cancel_all_cover_the_named_instruments() {
        let mut state = CommandState::default();
        let mut mock = Mock { positions: vec![7, 8], ..Mock::default() };
        let cmds = vec![Command::CloseAllFor("B".into()), Command::CancelAll];
        state.apply(&mut mock, cmds, 0, 4, 0).unwrap();
        assert_eq!(mock.log, ["close 1 7", "close 1 8", "cancel_all 0", "cancel_all 1"]);
    }

    #[test]
    fn held_intents_ride_the_next_step_ahead_of_new_ones() {
        let mut state = CommandState::default();
        let mut queue = Queue::default();
        queue.push_intent(Intent::Enter {
            size_frac: Some(0.5),
            stop_price: None,
            target_price: None,
        });
        state.hold(&mut queue, 3);
        assert!(state.has_held() && !queue.has_intents());
        // Another instrument's step does not see them.
        assert_eq!(state.fold(&mut queue, 1), Ok(Folded::default()));
        queue.push_intent(Intent::Enter {
            size_frac: Some(1.0),
            stop_price: None,
            target_price: None,
        });
        assert_eq!(state.fold(&mut queue, 3), Err(FoldError::DuplicateEntry));
    }

    #[test]
    fn fold_reports_duplicates_and_takes_the_entry_levels() {
        let mut state = CommandState::default();
        let mut queue = Queue::default();
        queue.push_intent(Intent::Enter {
            size_frac: Some(0.25),
            stop_price: Some(9.0),
            target_price: Some(12.0),
        });
        queue.push_intent(Intent::Close);
        let folded = state.fold(&mut queue, 0).unwrap();
        assert_eq!(
            folded,
            Folded {
                entry: true,
                exit: true,
                size_mult: Some(0.25),
                stop_price: Some(9.0),
                target_price: Some(12.0),
            }
        );
        queue.push_intent(Intent::Close);
        queue.push_intent(Intent::Close);
        let err = state.fold(&mut queue, 0).unwrap_err();
        assert_eq!(err.message("AAA event 4"), "duplicate close intents queued on AAA event 4");
    }

    #[test]
    fn repr_matches_python() {
        assert_eq!(py_repr("O-1"), "'O-1'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a\"b'c"), "'a\"b\\'c'");
        assert_eq!(py_repr("a\\b\n"), "'a\\\\b\\n'");
    }
}
