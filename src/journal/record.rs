//! One journal record and the kinds there are.

use super::value::Value;

/// Record kinds. A new kind only adds; a reader that does not know one keeps
/// it as data.
pub mod kind {
    /// The node was built: its shape, so a replay can tell it is the same node.
    pub const START: &str = "start";
    pub const TICK: &str = "tick";
    pub const BAR: &str = "bar";
    pub const DEPTH: &str = "depth";
    /// `advance_time`.
    pub const TIME: &str = "time";
    /// An execution report the venue sent.
    pub const REPORT: &str = "report";
    /// A batch of strategy commands, written before the engine acts on it.
    pub const COMMAND: &str = "command";
    /// The engine refused a batch of commands: the code it gave.
    pub const REFUSAL: &str = "refusal";
    /// A request for the adapter, written before the adapter is called.
    pub const ADAPTER_REQUEST: &str = "adapter_request";
    /// What the adapter call came to: ok, refused, unsupported, error, fenced,
    /// or one of the two outcomes a recovery cannot know.
    pub const ADAPTER_RESULT: &str = "adapter_result";
    pub const HALT: &str = "halt";
    pub const FINISH: &str = "finish";
    /// A note from the host (adoption, an operator action, ...).
    pub const ANNOTATION: &str = "annotation";
    /// A push that failed after it was journaled.
    pub const ERROR: &str = "error";
    /// Recovery: what the venue said, what the node expected.
    pub const RECONCILE: &str = "reconcile";
    pub const DISCREPANCY: &str = "discrepancy";
    /// Recovery: the host accepted the discrepancies and resumed.
    pub const RESUME: &str = "resume";
    pub const FENCE: &str = "fence";
    pub const HEARTBEAT: &str = "heartbeat";
    /// The venue restated or busted an execution and the book followed.
    pub const RESTATEMENT: &str = "restatement";
    /// A venue fill the order took and the book could not yet book (it is kept
    /// and booked when it can). Recovery reads these to know the book and the
    /// venue disagree about a position until the fill is booked.
    pub const UNBOOKED_FILL: &str = "unbooked_fill";
    /// The engine cut a working order's quantity (a sibling of a
    /// one-cancels-other group whose partner partly filled).
    pub const ORDER_RESIZED: &str = "order_resized";
    /// The engine closed a position by a rule of its own (stop, target, trail,
    /// liquidation, time exit, square-off).
    pub const ENGINE_EXIT: &str = "engine_exit";
    /// The book refused an order or an entry the engine had accepted from the
    /// strategy (position state, sizing, a limit): the outcome of a command.
    pub const REJECTION: &str = "rejection";
    /// A contract settled, or a corporate action was applied or refused.
    pub const LIFECYCLE: &str = "lifecycle";
    /// What the simulated venue (`PaperExecution`) answered, as a `report`
    /// record states a real venue's. Derived, never replayed as an input.
    pub const SIMULATED_REPORT: &str = "simulated_report";
    /// A position the account already held was given to the book
    /// (`initial_positions`, `adopt_position`).
    pub const ADOPTION: &str = "adoption";
    pub const STATE: &str = "state";
    /// An instrument was added to the running node: its symbol, spec, strategies and history.
    pub const INSTRUMENT_ADD: &str = "instrument_add";
    /// An instrument was retired from the running node.
    pub const INSTRUMENT_RETIRE: &str = "instrument_retire";
}

/// Who produced a record.
pub mod source {
    pub const FEED: &str = "feed";
    pub const VENUE: &str = "venue";
    pub const STRATEGY: &str = "strategy";
    pub const ENGINE: &str = "engine";
    pub const HOST: &str = "host";
}

/// A record as it is appended.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub seq: u64,
    pub kind: String,
    pub ts_event: i64,
    pub ts_arrival: i64,
    pub source: String,
    pub client_order_id: Option<String>,
    pub payload: Value,
}

impl Record {
    pub fn to_value(&self) -> Value {
        Value::map([
            ("seq", Value::Int(self.seq as i64)),
            ("kind", Value::str(&self.kind)),
            ("ts_event", Value::Int(self.ts_event)),
            ("ts_arrival", Value::Int(self.ts_arrival)),
            ("source", Value::str(&self.source)),
            ("client_order_id", Value::opt_str(self.client_order_id.as_deref())),
            ("payload", self.payload.clone()),
        ])
    }

    /// The canonical bytes the chain is taken over.
    pub fn canonical(&self) -> Vec<u8> {
        self.to_value().to_bytes()
    }

    pub fn from_value(v: &Value) -> Result<Record, String> {
        let int = |k: &str| {
            v.get(k).and_then(Value::as_int).ok_or_else(|| format!("record has no integer {k}"))
        };
        let string = |k: &str| {
            v.get(k)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("record has no string {k}"))
        };
        Ok(Record {
            seq: u64::try_from(int("seq")?).map_err(|_| "negative seq".to_string())?,
            kind: string("kind")?,
            ts_event: int("ts_event")?,
            ts_arrival: int("ts_arrival")?,
            source: string("source")?,
            client_order_id: match v.get("client_order_id") {
                Some(Value::Str(s)) => Some(s.clone()),
                _ => None,
            },
            payload: v.get("payload").cloned().unwrap_or(Value::Null),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_round_trips_through_its_canonical_bytes() {
        let r = Record {
            seq: 7,
            kind: kind::BAR.into(),
            ts_event: -5,
            ts_arrival: 9,
            source: source::FEED.into(),
            client_order_id: Some("s0-1".into()),
            payload: Value::map([("close", Value::Float(100.25))]),
        };
        let text = String::from_utf8(r.canonical()).unwrap();
        let back = Record::from_value(&Value::from_json(&text).unwrap()).unwrap();
        // The float came back as its text; the bytes are what matter.
        assert_eq!(back.canonical(), r.canonical());
        assert_eq!(back.seq, 7);
    }
}
