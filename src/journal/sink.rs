//! Where records go: a file, memory (held by the journal itself), or the
//! embedding application.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use super::chain::{chain_start, chain_step};
use super::core::Stored;
use super::record::{kind, Record};
use super::value::Value;

pub trait Sink {
    /// Take one record. `Err` means it is **not** durable.
    fn write(&mut self, stored: &Stored) -> Result<(), String>;
    /// Force everything written so far to stable storage.
    fn sync(&mut self) -> Result<(), String>;
}

/// When a file sink forces its records to stable storage. A record is always
/// handed to the operating system when it is appended, so a killed process
/// loses nothing under any policy; the policy decides what a power loss or an
/// operating-system crash can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fsync {
    /// After every record.
    Every,
    /// After every `n`th record.
    EveryN(u32),
    /// After the records that precede or follow an order leaving the node:
    /// strategy commands, adapter requests and results, reports, halts,
    /// refusals and the finish. Market data is not forced.
    Orders,
    /// Never; the operating system flushes when it chooses.
    Never,
}

fn forces(kind_name: &str) -> bool {
    matches!(
        kind_name,
        kind::COMMAND
            | kind::REFUSAL
            | kind::ADAPTER_REQUEST
            | kind::ADAPTER_RESULT
            | kind::REPORT
            | kind::HALT
            | kind::FINISH
            | kind::RESUME
            | kind::RESTATEMENT
            | kind::UNBOOKED_FILL
            | kind::ORDER_RESIZED
            | kind::ADOPTION
            | kind::ENGINE_EXIT
            | kind::LIFECYCLE
            | kind::REJECTION
            | kind::SIMULATED_REPORT
            | kind::INSTRUMENT_ADD
            | kind::INSTRUMENT_RETIRE
    )
}

const MAGIC: &[u8; 8] = b"RBTJRNL1";
/// Frames are `u32 length, u32 crc32 (both little endian), canonical bytes`.
const FRAME_HEAD: usize = 8;

pub struct FileSink {
    file: File,
    policy: Fsync,
    since_sync: u32,
}

fn frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEAD + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

impl FileSink {
    /// Create `path` with the journal's header, or reopen it to append after
    /// the records already there (the caller has read and verified them with
    /// [`read_file`]; a torn tail is cut here).
    pub fn open(
        path: &Path,
        policy: Fsync,
        header: &Value,
        seq_base: u64,
        chain_from: Option<&str>,
    ) -> Result<FileSink, String> {
        let existing = std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if existing {
            let contents = read_file(path)?;
            if contents.torn_bytes > 0 {
                file.set_len(contents.valid_len).map_err(|e| e.to_string())?;
            }
        } else {
            let head = Value::map([
                ("header", header.clone()),
                ("seq_base", Value::Int(seq_base as i64)),
                ("chain_from", Value::opt_str(chain_from)),
            ]);
            let mut bytes = MAGIC.to_vec();
            bytes.extend(frame(&head.to_bytes()));
            file.write_all(&bytes).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
        }
        Ok(FileSink { file, policy, since_sync: 0 })
    }
}

impl Sink for FileSink {
    fn write(&mut self, stored: &Stored) -> Result<(), String> {
        self.file.write_all(&frame(&stored.canonical)).map_err(|e| e.to_string())?;
        let force = match self.policy {
            Fsync::Every => true,
            Fsync::EveryN(n) => {
                self.since_sync += 1;
                self.since_sync >= n.max(1)
            }
            Fsync::Orders => forces(&stored.record.kind),
            Fsync::Never => false,
        };
        if force {
            self.sync()?;
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<(), String> {
        self.since_sync = 0;
        self.file.sync_data().map_err(|e| e.to_string())
    }
}

/// What a journal file holds.
pub struct FileContents {
    pub header: Value,
    pub seq_base: u64,
    pub chain_from: Option<String>,
    pub records: Vec<Stored>,
    /// Bytes after the last whole, checksummed frame: the write a crash cut.
    pub torn_bytes: u64,
    pub valid_len: u64,
}

/// Read a journal file, verifying every checksum and the chain. A truncated or
/// corrupt **last** frame is reported as torn bytes (a crash mid-write); a bad
/// frame with whole frames after it is an error, since nothing a crash does
/// makes that.
pub fn read_file(path: &Path) -> Result<FileContents, String> {
    let mut data = Vec::new();
    File::open(path)
        .and_then(|mut f| f.read_to_end(&mut data))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    parse_file(&data)
}

/// Whether some whole, checksummed, non-empty frame starts anywhere in `bytes`. Payloads are
/// canonical JSON text, so a stray offset reads a length no file of this kind reaches and is
/// skipped without hashing anything.
fn whole_frame_in(bytes: &[u8]) -> bool {
    (0..bytes.len().saturating_sub(FRAME_HEAD)).any(|i| {
        let rest = &bytes[i..];
        let len = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(rest[4..8].try_into().unwrap());
        len > 0
            && rest.len() >= FRAME_HEAD + len
            && crc32fast::hash(&rest[FRAME_HEAD..FRAME_HEAD + len]) == crc
    })
}

pub fn parse_file(data: &[u8]) -> Result<FileContents, String> {
    if data.len() < MAGIC.len() || &data[..MAGIC.len()] != MAGIC {
        return Err("not a raptorbt journal (bad magic)".into());
    }
    let mut pos = MAGIC.len();
    let mut frames: Vec<&[u8]> = Vec::new();
    let mut torn = 0u64;
    while pos < data.len() {
        let rest = &data[pos..];
        if rest.len() < FRAME_HEAD {
            torn = rest.len() as u64;
            break;
        }
        let len = u32::from_le_bytes(rest[0..4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(rest[4..8].try_into().unwrap());
        if rest.len() < FRAME_HEAD + len {
            // A crash leaves a short last frame with nothing after it. A length field that
            // overruns the file while whole frames sit behind it is damage, not a crash: refuse
            // it rather than cut the records away.
            if whole_frame_in(&rest[FRAME_HEAD.min(rest.len())..]) {
                return Err(format!(
                    "corrupt record at byte {pos}: its length overruns the file but whole frames follow"
                ));
            }
            torn = rest.len() as u64;
            break;
        }
        let payload = &rest[FRAME_HEAD..FRAME_HEAD + len];
        if crc32fast::hash(payload) != crc {
            if pos + FRAME_HEAD + len == data.len() {
                torn = rest.len() as u64;
                break;
            }
            return Err(format!("corrupt record at byte {pos}: checksum mismatch"));
        }
        frames.push(payload);
        pos += FRAME_HEAD + len;
    }
    let valid_len = pos as u64;
    let Some((head, body)) = frames.split_first() else {
        return Err("journal file has no header".into());
    };
    let head = Value::from_json(std::str::from_utf8(head).map_err(|e| e.to_string())?)?;
    let header = head.get("header").cloned().ok_or("journal header frame has no header")?;
    let seq_base = head.get("seq_base").and_then(Value::as_int).ok_or("no seq_base")? as u64;
    let chain_from = head.get("chain_from").and_then(Value::as_str).map(str::to_string);
    let mut chain = chain_from.clone().unwrap_or_else(|| chain_start(&header.to_bytes()));
    let mut records = Vec::with_capacity(body.len());
    for (i, payload) in body.iter().enumerate() {
        let text = std::str::from_utf8(payload).map_err(|e| e.to_string())?;
        let record = Record::from_value(&Value::from_json(text)?)?;
        if record.seq != seq_base + i as u64 {
            return Err(format!(
                "record {i} has seq {}, expected {}",
                record.seq,
                seq_base + i as u64
            ));
        }
        chain = chain_step(&chain, payload);
        records.push(Stored { record, canonical: payload.to_vec(), chain: chain.clone() });
    }
    Ok(FileContents { header, seq_base, chain_from, records, torn_bytes: torn, valid_len })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::core::Journal;

    fn temp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("rbt-journal-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn fill(j: &mut Journal, n: i64) {
        for i in 0..n {
            j.append(kind::BAR, i, i, "feed", None, Value::map([("c", Value::Float(i as f64))]))
                .unwrap();
        }
    }

    #[test]
    fn the_orders_policy_forces_every_record_that_changes_an_order_or_a_position() {
        for k in [
            kind::COMMAND,
            kind::REFUSAL,
            kind::REJECTION,
            kind::SIMULATED_REPORT,
            kind::ORDER_RESIZED,
            kind::ENGINE_EXIT,
            kind::LIFECYCLE,
            kind::ADOPTION,
            kind::RESTATEMENT,
            kind::UNBOOKED_FILL,
            kind::HALT,
        ] {
            assert!(forces(k), "{k} must reach stable storage under the orders policy");
        }
        assert!(!forces(kind::BAR) && !forces(kind::TICK));
    }

    #[test]
    fn a_file_reads_back_with_the_same_chain() {
        let path = temp("rt");
        let header = Value::map([("run", Value::str("x"))]);
        let mut j = Journal::new(header.clone(), 0, None);
        j.add_sink(Box::new(FileSink::open(&path, Fsync::Every, &header, 0, None).unwrap()));
        fill(&mut j, 20);
        let got = read_file(&path).unwrap();
        assert_eq!(got.records.len(), 20);
        assert_eq!(got.records.last().unwrap().chain, j.chain());
        assert_eq!(got.torn_bytes, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_cut_tail_is_torn_and_a_flipped_byte_in_the_middle_is_an_error() {
        let path = temp("torn");
        let header = Value::Null;
        let mut j = Journal::new(header.clone(), 0, None);
        j.add_sink(Box::new(FileSink::open(&path, Fsync::Never, &header, 0, None).unwrap()));
        fill(&mut j, 5);
        let data = std::fs::read(&path).unwrap();
        for cut in 1..40 {
            let got = parse_file(&data[..data.len() - cut]).unwrap();
            assert_eq!(got.records.len(), 4);
            assert!(got.torn_bytes > 0);
        }
        let mut bad = data.clone();
        bad[MAGIC.len() + FRAME_HEAD + 30] ^= 0x55; // inside the first record
        assert!(parse_file(&bad).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reopening_cuts_the_torn_tail_and_appends_after_it() {
        let path = temp("reopen");
        let header = Value::Null;
        let mut j = Journal::new(header.clone(), 0, None);
        j.add_sink(Box::new(FileSink::open(&path, Fsync::Never, &header, 0, None).unwrap()));
        fill(&mut j, 3);
        let mut data = std::fs::read(&path).unwrap();
        data.extend_from_slice(&[9, 0, 0]); // a frame header cut short
        std::fs::write(&path, &data).unwrap();
        let mut again = FileSink::open(&path, Fsync::Never, &header, 0, None).unwrap();
        let last = read_file(&path).unwrap();
        assert_eq!(last.torn_bytes, 0);
        assert_eq!(last.records.len(), 3);
        let _ = &mut again;
        let _ = std::fs::remove_file(&path);
    }
}
