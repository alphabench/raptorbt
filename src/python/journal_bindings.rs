//! `raptorbt.journal`: the journal, its records and the chain function.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use super::errors::Fail;
use crate::journal::sink::{read_file, FileSink, Fsync, Sink};
use crate::journal::{chain_start, chain_step, Journal, JournalError, Record, Stored, Value};

/// The error a journal failure raises.
pub fn journal_err(e: JournalError) -> PyErr {
    match e {
        JournalError::Diverged { .. } | JournalError::Replay(_) => {
            Fail::new("replay_diverged", e.to_string()).into()
        }
        JournalError::Evicted { wanted, oldest } => Fail::new("journal_evicted", e.to_string())
            .param("seq")
            .value(wanted as i64)
            .valid([oldest])
            .into(),
        JournalError::Sink(_) | JournalError::Broken => {
            Fail::new("journal_failed", e.to_string()).into()
        }
    }
}

/// A Python value as a record value. Floats are kept as floats here and
/// written as text by the encoder.
pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Value::Float(obj.extract::<f64>()?));
    }
    if obj.is_instance_of::<PyInt>() {
        return obj.extract::<i64>().map(Value::Int).map_err(|_| {
            Fail::new("invalid_parameter", "a journal integer must fit in 64 bits").into()
        });
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::Str(s.to_str()?.to_string()));
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(Value::Str(b.as_bytes().iter().map(|x| format!("{x:02x}")).collect()));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut map = std::collections::BTreeMap::new();
        for (k, v) in d.iter() {
            let key = k.cast::<PyString>().map_err(|_| {
                PyErr::from(Fail::new("invalid_parameter", "a journal map has text keys"))
            })?;
            map.insert(key.to_str()?.to_string(), py_to_value(&v)?);
        }
        return Ok(Value::Map(map));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return Ok(Value::List(
            obj.try_iter()?.map(|i| py_to_value(&i?)).collect::<PyResult<_>>()?,
        ));
    }
    // numpy scalars and other number types
    if let Ok(i) = obj.call_method0("__index__").and_then(|x| x.extract::<i64>()) {
        return Ok(Value::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Float(f));
    }
    Err(Fail::new(
        "invalid_parameter",
        format!("cannot write a {} into a journal record", obj.get_type().name()?),
    )
    .into())
}

/// A record value as Python: floats are the strings they were written as.
pub fn value_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Float(f) => crate::journal::value::float_text(*f).into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Map(m) => {
            let d = PyDict::new(py);
            for (k, item) in m {
                d.set_item(k, value_to_py(py, item)?)?;
            }
            d.into_any()
        }
    })
}

/// One record, as read back.
#[pyclass(name = "Record", module = "raptorbt.journal", unsendable)]
pub struct PyRecord {
    pub stored: Rc<Stored>,
}

#[pymethods]
impl PyRecord {
    #[getter]
    fn seq(&self) -> u64 {
        self.stored.record.seq
    }
    #[getter]
    fn ts_event(&self) -> i64 {
        self.stored.record.ts_event
    }
    #[getter]
    fn ts_arrival(&self) -> i64 {
        self.stored.record.ts_arrival
    }
    #[getter]
    fn kind(&self) -> &str {
        &self.stored.record.kind
    }
    #[getter]
    fn source(&self) -> &str {
        &self.stored.record.source
    }
    #[getter]
    fn client_order_id(&self) -> Option<&str> {
        self.stored.record.client_order_id.as_deref()
    }
    /// The chain head after this record.
    #[getter]
    fn chain(&self) -> &str {
        &self.stored.chain
    }
    /// The payload as plain Python values. Floats are decimal strings (the
    /// record's bytes carry no float).
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        value_to_py(py, &self.stored.record.payload)
    }
    /// The bytes the chain was taken over.
    fn canonical<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.stored.canonical)
    }
    fn __repr__(&self) -> String {
        let r = &self.stored.record;
        format!(
            "Record(seq={}, kind={:?}, ts_event={}, source={:?})",
            r.seq, r.kind, r.ts_event, r.source
        )
    }
}

struct CallbackSink(Py<PyAny>);

impl Sink for CallbackSink {
    fn write(&mut self, stored: &Stored) -> Result<(), String> {
        Python::attach(|py| {
            let rec = Py::new(py, PyRecord { stored: Rc::new(stored.clone()) })
                .map_err(|e| e.to_string())?;
            self.0.call1(py, (rec,)).map(|_| ()).map_err(|e| e.value(py).to_string())
        })
    }

    fn sync(&mut self) -> Result<(), String> {
        Ok(())
    }
}

fn parse_fsync(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Fsync> {
    let Some(obj) = obj else { return Ok(Fsync::Orders) };
    if obj.is_none() {
        return Ok(Fsync::Orders);
    }
    if let Ok(n) = obj.extract::<u32>() {
        if !obj.is_instance_of::<PyBool>() && n >= 1 {
            return Ok(Fsync::EveryN(n));
        }
    }
    match obj.extract::<String>().ok().as_deref() {
        Some("every") => Ok(Fsync::Every),
        Some("orders") => Ok(Fsync::Orders),
        Some("never") => Ok(Fsync::Never),
        _ => Err(Fail::new("invalid_parameter", "fsync is 'every', 'orders', 'never' or a count")
            .param("fsync")
            .valid(["every", "orders", "never", "N"])
            .into()),
    }
}

/// `(header, seq_base, chain_from, records, torn_bytes)`.
type JournalFile<'py> = (Bound<'py, PyAny>, u64, Option<String>, Vec<Py<PyRecord>>, u64);

/// The journal of a node. Give it to `LiveNode(journal=...)`.
#[pyclass(name = "Journal", module = "raptorbt.journal", unsendable)]
pub struct PyJournal {
    pub inner: Rc<RefCell<Journal>>,
}

#[pymethods]
impl PyJournal {
    /// `header` is opaque to the engine: whatever the host needs to say what
    /// produced the run. `seq_base` and `chain` continue an earlier run: the
    /// first record gets sequence number `seq_base` and the chain goes on from
    /// `chain` instead of starting at the header. `file` is a path (created, or
    /// reopened to append after verifying what is there); `fsync` is `"every"`,
    /// `"orders"` (the default), `"never"` or a count N (every Nth record).
    /// `memory` keeps records for `since`: `True` for all, an int for the
    /// latest N. `sink` is called with each `Record` before the engine acts on
    /// what it describes; an exception from it stops the node (the record is
    /// not durable).
    #[new]
    #[pyo3(signature = (header=None, *, seq_base=0, chain=None, file=None, fsync=None, memory=None, sink=None))]
    fn new(
        header: Option<&Bound<'_, PyAny>>,
        seq_base: u64,
        chain: Option<String>,
        file: Option<PathBuf>,
        fsync: Option<&Bound<'_, PyAny>>,
        memory: Option<&Bound<'_, PyAny>>,
        sink: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        (|| -> PyResult<Self> {
            let header = match header {
                Some(h) => py_to_value(h)?,
                None => Value::Null,
            };
            let policy = parse_fsync(fsync)?;
            let mut j = Journal::new(header.clone(), seq_base, chain.clone());
            if let Some(path) = &file {
                let sink = FileSink::open(path, policy, &header, seq_base, chain.as_deref())
                    .map_err(|e| PyErr::from(Fail::new("journal_failed", e)))?;
                j.add_byte_sink(Box::new(sink));
            }
            if let Some(cb) = sink {
                j.add_sink(Box::new(CallbackSink(cb)));
            }
            if let Some(m) = memory {
                if let Ok(b) = m.cast::<PyBool>() {
                    if b.is_true() {
                        j.keep_in_memory(None);
                    }
                } else if !m.is_none() {
                    j.keep_in_memory(Some(m.extract::<usize>()?.max(1)));
                }
            }
            Ok(PyJournal { inner: Rc::new(RefCell::new(j)) })
        })()
        .map_err(|e| super::errors::with_op(e, "Journal"))
    }

    /// Records with `seq >= seq`, oldest first. Raises `journal_evicted` when
    /// a bounded memory has dropped some of them.
    fn since(&self, py: Python<'_>, seq: u64) -> PyResult<Vec<Py<PyRecord>>> {
        let records = self.inner.borrow().since(seq).map_err(journal_err)?;
        records.into_iter().map(|stored| Py::new(py, PyRecord { stored })).collect()
    }

    /// `(count, chain, header)`: how many records the chain holds (also the
    /// sequence number of the next one), the chain head, and the header.
    fn head<'py>(&self, py: Python<'py>) -> PyResult<(u64, String, Bound<'py, PyAny>)> {
        let j = self.inner.borrow();
        Ok((j.count(), j.chain().to_string(), value_to_py(py, j.header())?))
    }

    #[getter]
    fn count(&self) -> u64 {
        self.inner.borrow().count()
    }
    #[getter]
    fn chain(&self) -> String {
        self.inner.borrow().chain().to_string()
    }
    /// The chain before the first record.
    #[getter]
    fn chain_from(&self) -> String {
        self.inner.borrow().chain_from().to_string()
    }
    #[getter]
    fn seq_base(&self) -> u64 {
        self.inner.borrow().seq_base()
    }
    #[getter]
    fn broken(&self) -> bool {
        self.inner.borrow().is_broken()
    }
    #[getter]
    fn last_kind(&self) -> Option<String> {
        self.inner.borrow().last_kind().map(str::to_string)
    }

    /// Append a record from the host. `payload` is plain values (dict, list,
    /// str, int, float, bool, None); floats are stored as text.
    #[pyo3(signature = (kind, ts_event, payload=None, *, ts_arrival=None, source="host", client_order_id=None))]
    fn append(
        &self,
        kind: &str,
        ts_event: i64,
        payload: Option<&Bound<'_, PyAny>>,
        ts_arrival: Option<i64>,
        source: &str,
        client_order_id: Option<&str>,
    ) -> PyResult<u64> {
        let payload = match payload {
            Some(p) => py_to_value(p)?,
            None => Value::Null,
        };
        self.inner
            .borrow_mut()
            .append(
                kind,
                ts_event,
                ts_arrival.unwrap_or(ts_event),
                source,
                client_order_id,
                payload,
            )
            .map_err(journal_err)
    }

    /// Force every sink to stable storage now.
    fn sync(&self) -> PyResult<()> {
        self.inner.borrow_mut().sync().map_err(journal_err)
    }

    /// Make the next records appended exactly these (a replay). `records` are
    /// `Record`s or their `canonical()` bytes, in order, starting at this
    /// journal's `count`.
    fn expect(&self, records: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let mut j = self.inner.borrow_mut();
        let mut chain = j.chain().to_string();
        let mut out = Vec::with_capacity(records.len());
        for (next, r) in (j.count()..).zip(records) {
            let stored = if let Ok(rec) = r.cast::<PyRecord>() {
                rec.borrow().stored.clone()
            } else {
                let bytes: Vec<u8> = r.extract()?;
                let text = std::str::from_utf8(&bytes)
                    .map_err(|e| PyErr::from(Fail::new("invalid_checkpoint", e.to_string())))?;
                let record = Value::from_json(text)
                    .and_then(|v| Record::from_value(&v))
                    .map_err(|e| PyErr::from(Fail::new("invalid_checkpoint", e)))?;
                Rc::new(Stored { record, canonical: bytes, chain: String::new() })
            };
            if stored.record.seq != next {
                return Err(Fail::new(
                    "replay_diverged",
                    format!(
                        "record {} where {next} was expected: the journal has a gap",
                        stored.record.seq
                    ),
                )
                .into());
            }
            chain = chain_step(&chain, &stored.canonical);
            if !stored.chain.is_empty() && stored.chain != chain {
                return Err(Fail::new(
                    "replay_diverged",
                    format!("record {next} does not continue the chain"),
                )
                .into());
            }
            out.push(Rc::new(Stored {
                record: stored.record.clone(),
                canonical: stored.canonical.clone(),
                chain: chain.clone(),
            }));
        }
        j.expect(out);
        Ok(())
    }

    /// Whether a replay is still rebuilding the node.
    #[getter]
    fn replaying(&self) -> bool {
        self.inner.borrow().is_replaying()
    }

    /// Records a replay has still to reproduce.
    #[getter]
    fn remaining(&self) -> usize {
        self.inner.borrow().remaining()
    }

    /// The next record a replay expects, if any.
    fn peek(&self, py: Python<'_>) -> PyResult<Option<Py<PyRecord>>> {
        let j = self.inner.borrow();
        match j.peek_expected() {
            Some(s) => Ok(Some(Py::new(py, PyRecord { stored: Rc::new(s.clone()) })?)),
            None => Ok(None),
        }
    }

    /// Read a journal file: `(header, seq_base, records, torn_bytes)`. Every
    /// checksum and the chain are verified; a cut last record is reported in
    /// `torn_bytes`, not raised.
    #[staticmethod]
    fn read<'py>(py: Python<'py>, path: PathBuf) -> PyResult<JournalFile<'py>> {
        let c = read_file(&path).map_err(|e| PyErr::from(Fail::new("journal_failed", e)))?;
        let records = c
            .records
            .into_iter()
            .map(|s| Py::new(py, PyRecord { stored: Rc::new(s) }))
            .collect::<PyResult<Vec<_>>>()?;
        Ok((value_to_py(py, &c.header)?, c.seq_base, c.chain_from, records, c.torn_bytes))
    }
}

/// `chain_step(prev_hex, canonical) -> hex`: one link of the hash chain.
/// Pure. `canonical` is a record's `canonical()` bytes.
#[pyfunction]
#[pyo3(name = "chain_step")]
pub fn py_chain_step(prev_hex: &str, canonical: &[u8]) -> String {
    chain_step(prev_hex, canonical)
}

/// The chain's first link for a run described by `header` (a dict).
#[pyfunction]
#[pyo3(name = "chain_start")]
pub fn py_chain_start(header: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(chain_start(&py_to_value(header)?.to_bytes()))
}

/// The canonical bytes of plain Python values, as a record would write them.
#[pyfunction]
#[pyo3(name = "canonical")]
pub fn py_canonical<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(py, &py_to_value(value)?.to_bytes()))
}

pub fn missing(key: &str) -> PyErr {
    PyKeyError::new_err(key.to_string())
}
