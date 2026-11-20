//! The journal: an append-only, hash-chained record of everything a node was
//! told and everything it did.
//!
//! A record is written **before** the engine acts on what it describes (an
//! input before it is applied, a request before the adapter is called), so a
//! process that dies at any point leaves a journal from which the state and
//! the unanswered requests can be rebuilt.

use std::collections::VecDeque;
use std::rc::Rc;

use super::chain::{chain_start, chain_step};
use super::market::Payload;
use super::record::{kind, Record};
use super::sink::Sink;
use super::value::Value;

/// A record with its bytes and its place in the chain.
#[derive(Debug, Clone)]
pub struct Stored {
    pub record: Record,
    pub canonical: Vec<u8>,
    /// The chain head after this record.
    pub chain: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JournalError {
    /// A sink could not take the record. The journal is closed to further
    /// appends: a gap in a hash chain cannot be repaired afterwards.
    Sink(String),
    /// An earlier append failed.
    Broken,
    /// While replaying, the node produced a record that is not the one the
    /// journal holds at that place.
    Diverged { seq: u64, expected: String, got: String },
    /// Replay ended mid-step or the journal holds records the replay did not
    /// produce.
    Replay(String),
    /// A read asked for records that have been evicted from memory.
    Evicted { wanted: u64, oldest: u64 },
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JournalError::Sink(m) => write!(f, "the journal could not write a record: {m}"),
            JournalError::Broken => write!(f, "the journal is closed after a failed write"),
            JournalError::Diverged { seq, expected, got } => write!(
                f,
                "replay diverged at record {seq}: the journal holds {expected}, the node produced {got}"
            ),
            JournalError::Replay(m) => write!(f, "{m}"),
            JournalError::Evicted { wanted, oldest } => write!(
                f,
                "record {wanted} is no longer held in memory (the oldest held is {oldest}); read it from the durable sink"
            ),
        }
    }
}

impl std::error::Error for JournalError {}

/// Records kept in memory for read-back.
struct Memory {
    cap: Option<usize>,
    held: VecDeque<Rc<Stored>>,
}

pub struct Journal {
    header: Value,
    header_canonical: Vec<u8>,
    seq_base: u64,
    next_seq: u64,
    /// The chain before the first record.
    chain_from: String,
    chain: String,
    sinks: Vec<Box<dyn Sink>>,
    /// Sinks that may read a record's value, not only its bytes.
    value_sinks: usize,
    memory: Option<Memory>,
    /// Records a replay must reproduce, front first.
    expected: VecDeque<Rc<Stored>>,
    replayed: u64,
    broken: bool,
    last_kind: Option<String>,
    /// The buffer the next record's bytes are written into, when the last
    /// record's were not kept.
    scratch: String,
}

impl Journal {
    /// A journal for a run described by `header`. `seq_base` is the sequence
    /// number of its first record, and `chain`, when given, is the head of the
    /// run it continues (otherwise the chain starts from the header).
    pub fn new(header: Value, seq_base: u64, chain: Option<String>) -> Self {
        let header_canonical = header.to_bytes();
        let chain = chain.unwrap_or_else(|| chain_start(&header_canonical));
        Journal {
            header,
            header_canonical,
            seq_base,
            next_seq: seq_base,
            chain_from: chain.clone(),
            chain,
            sinks: Vec::new(),
            value_sinks: 0,
            memory: None,
            expected: VecDeque::new(),
            replayed: 0,
            broken: false,
            last_kind: None,
            scratch: String::new(),
        }
    }

    pub fn add_sink(&mut self, sink: Box<dyn Sink>) {
        self.value_sinks += 1;
        self.sinks.push(sink);
    }

    /// A sink that reads only a record's bytes and its kind, never its other
    /// fields: while every sink is one and no memory is kept, a market-data
    /// record reaches the sinks with a `Null` payload and its bytes written
    /// without building the value (see [`Journal::append_payload`]).
    pub(crate) fn add_byte_sink(&mut self, sink: Box<dyn Sink>) {
        self.sinks.push(sink);
    }

    /// Keep records in memory for [`Journal::since`]: the latest `cap`, or all.
    pub fn keep_in_memory(&mut self, cap: Option<usize>) {
        self.memory = Some(Memory { cap, held: VecDeque::new() });
    }

    pub fn header(&self) -> &Value {
        &self.header
    }

    pub fn header_canonical(&self) -> &[u8] {
        &self.header_canonical
    }

    pub fn seq_base(&self) -> u64 {
        self.seq_base
    }

    /// Sequence number the next record gets, which is also the number of
    /// records in the chain so far.
    pub fn count(&self) -> u64 {
        self.next_seq
    }

    pub fn chain(&self) -> &str {
        &self.chain
    }

    /// The chain before the first record: the header's hash, or the head of
    /// the run this one continues.
    pub fn chain_from(&self) -> &str {
        &self.chain_from
    }

    pub fn last_kind(&self) -> Option<&str> {
        self.last_kind.as_deref()
    }

    pub fn is_broken(&self) -> bool {
        self.broken
    }

    /// Begin a replay: the next records appended must be exactly these.
    pub fn expect(&mut self, records: Vec<Rc<Stored>>) {
        self.expected = records.into();
    }

    pub fn is_replaying(&self) -> bool {
        !self.expected.is_empty()
    }

    pub fn peek_expected(&self) -> Option<&Stored> {
        self.expected.front().map(|r| &**r)
    }

    /// Records still to be reproduced.
    pub fn remaining(&self) -> usize {
        self.expected.len()
    }

    /// How many records a replay has reproduced.
    pub fn replayed(&self) -> u64 {
        self.replayed
    }

    /// Append a record. While replaying, the record must equal the one the
    /// journal holds next, and nothing is written (it already is).
    pub fn append(
        &mut self,
        kind: &str,
        ts_event: i64,
        ts_arrival: i64,
        source: &str,
        client_order_id: Option<&str>,
        payload: Value,
    ) -> Result<u64, JournalError> {
        self.append_with(kind, ts_event, ts_arrival, source, client_order_id, payload, None)
    }

    /// Append a record of market data. Its bytes are those [`Journal::append`]
    /// writes for `payload.value()`; the value is built only when something
    /// keeps it (memory, or a sink other than a byte sink).
    pub(crate) fn append_payload(
        &mut self,
        kind: &str,
        ts_event: i64,
        ts_arrival: i64,
        source: &str,
        payload: &impl Payload,
    ) -> Result<u64, JournalError> {
        if self.memory.is_some() || self.value_sinks > 0 {
            return self.append(kind, ts_event, ts_arrival, source, None, payload.value());
        }
        let write = |out: &mut String| payload.write(out);
        self.append_with(kind, ts_event, ts_arrival, source, None, Value::Null, Some(&write))
    }

    /// `write`, when given, writes the payload's bytes in place of `payload`.
    #[allow(clippy::too_many_arguments)]
    fn append_with(
        &mut self,
        kind: &str,
        ts_event: i64,
        ts_arrival: i64,
        source: &str,
        client_order_id: Option<&str>,
        payload: Value,
        write: Option<&dyn Fn(&mut String)>,
    ) -> Result<u64, JournalError> {
        if self.broken {
            return Err(JournalError::Broken);
        }
        let record = Record {
            seq: self.next_seq,
            kind: kind.to_string(),
            ts_event,
            ts_arrival,
            source: source.to_string(),
            client_order_id: client_order_id.map(str::to_string),
            payload,
        };
        let mut text = std::mem::take(&mut self.scratch);
        text.clear();
        // A record kept in memory takes its buffer with it: the next one
        // starts from the room `Value::to_bytes` gives.
        text.reserve(128);
        match write {
            Some(write) => record.write_canonical(&mut text, write),
            None => record.write_canonical(&mut text, |out| record.payload.write(out)),
        }
        let canonical = text.into_bytes();
        let seq = record.seq;
        if let Some(exp) = self.expected.front() {
            if exp.canonical != canonical {
                return Err(JournalError::Diverged {
                    seq,
                    expected: String::from_utf8_lossy(&exp.canonical).into_owned(),
                    got: String::from_utf8_lossy(&canonical).into_owned(),
                });
            }
            let stored = self.expected.pop_front().expect("peeked");
            self.replayed += 1;
            self.reuse(canonical);
            self.advance(stored);
            return Ok(seq);
        }
        let chain = chain_step(&self.chain, &canonical);
        let stored = Stored { record, canonical, chain };
        for sink in &mut self.sinks {
            if let Err(e) = sink.write(&stored) {
                self.broken = true;
                return Err(JournalError::Sink(e));
            }
        }
        if self.memory.is_some() {
            self.advance(Rc::new(stored));
        } else {
            // Nothing keeps the record: its link becomes the head and its
            // buffer takes the next record's bytes.
            self.next_seq = seq + 1;
            self.set_last_kind(kind);
            self.chain = stored.chain;
            self.reuse(stored.canonical);
        }
        Ok(seq)
    }

    fn reuse(&mut self, mut bytes: Vec<u8>) {
        bytes.clear();
        self.scratch = String::from_utf8(bytes).expect("an empty buffer is text");
    }

    fn set_last_kind(&mut self, kind: &str) {
        if let Some(last) = &mut self.last_kind {
            last.clear();
            last.push_str(kind);
        } else {
            self.last_kind = Some(kind.to_string());
        }
    }

    fn advance(&mut self, stored: Rc<Stored>) {
        self.next_seq = stored.record.seq + 1;
        self.chain.clear();
        self.chain.push_str(&stored.chain);
        self.set_last_kind(&stored.record.kind);
        if let Some(mem) = &mut self.memory {
            mem.held.push_back(stored);
            if let Some(cap) = mem.cap {
                while mem.held.len() > cap {
                    mem.held.pop_front();
                }
            }
        }
    }

    /// Records with `seq >= from`, oldest first.
    pub fn since(&self, from: u64) -> Result<Vec<Rc<Stored>>, JournalError> {
        let Some(mem) = &self.memory else {
            return Err(JournalError::Replay(
                "this journal keeps no records in memory; give it memory= or read its file".into(),
            ));
        };
        let oldest = mem.held.front().map_or(self.next_seq, |s| s.record.seq);
        if from < oldest && oldest > self.seq_base {
            return Err(JournalError::Evicted { wanted: from, oldest });
        }
        Ok(mem.held.iter().filter(|s| s.record.seq >= from).cloned().collect())
    }

    /// Flush every sink to stable storage.
    pub fn sync(&mut self) -> Result<(), JournalError> {
        for sink in &mut self.sinks {
            sink.sync().map_err(JournalError::Sink)?;
        }
        Ok(())
    }

    /// Write one `error` record for a push that failed after it was journaled.
    pub fn note_error(&mut self, ts: i64, op: &str, code: &str, message: &str) {
        let payload = Value::map([
            ("op", Value::str(op)),
            ("code", Value::str(code)),
            ("message", Value::str(message)),
        ]);
        // A failure to journal the failure surfaces on the next append.
        let _ = self.append(kind::ERROR, ts, ts, super::record::source::ENGINE, None, payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn j() -> Journal {
        let mut j = Journal::new(Value::map([("run", Value::str("r1"))]), 0, None);
        j.keep_in_memory(None);
        j
    }

    fn put(j: &mut Journal, n: i64) -> u64 {
        j.append(kind::BAR, n, n, "feed", None, Value::map([("c", Value::Float(n as f64))]))
            .unwrap()
    }

    #[test]
    fn same_inputs_same_chain_head() {
        let (mut a, mut b) = (j(), j());
        for n in 0..50 {
            put(&mut a, n);
            put(&mut b, n);
        }
        assert_eq!(a.chain(), b.chain());
        assert_eq!(a.count(), 50);
        put(&mut b, 99);
        assert_ne!(a.chain(), b.chain());
    }

    #[test]
    fn a_continuation_carries_the_chain_and_the_sequence() {
        let mut a = j();
        for n in 0..5 {
            put(&mut a, n);
        }
        let mut b = Journal::new(
            Value::map([("run", Value::str("r2"))]),
            a.count(),
            Some(a.chain().to_string()),
        );
        assert_eq!(put(&mut b, 5), 5);
        put(&mut a, 5);
        assert_eq!(a.chain(), b.chain());
    }

    #[test]
    fn replay_checks_each_record_and_stops_at_the_first_difference() {
        let mut a = j();
        for n in 0..4 {
            put(&mut a, n);
        }
        let kept = a.since(0).unwrap();
        let mut r = j();
        r.expect(kept);
        for n in 0..3 {
            put(&mut r, n);
        }
        assert!(r.is_replaying());
        let err = r.append(kind::BAR, 3, 3, "feed", None, Value::map([("c", Value::Float(7.0))]));
        assert!(matches!(err, Err(JournalError::Diverged { seq: 3, .. })));
        // The failed append changed nothing.
        assert_eq!(r.count(), 3);
        put(&mut r, 3);
        assert!(!r.is_replaying());
        assert_eq!(r.chain(), a.chain());
    }

    #[test]
    fn memory_is_bounded_and_says_so_when_asked_for_what_it_dropped() {
        let mut a = Journal::new(Value::Null, 0, None);
        a.keep_in_memory(Some(3));
        for n in 0..10 {
            put(&mut a, n);
        }
        assert_eq!(a.since(7).unwrap().len(), 3);
        assert!(matches!(a.since(2), Err(JournalError::Evicted { wanted: 2, oldest: 7 })));
    }
}
