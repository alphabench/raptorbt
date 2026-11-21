//! The journal: what a node was told, what it decided and what it sent, in an
//! append-only hash chain. See
//! `docs/architecture/journal-and-recovery.md`.

pub mod adapter;
pub mod chain;
pub mod core;
pub(crate) mod market;
pub mod record;
pub mod sink;
pub mod value;

pub use self::core::{Journal, JournalError, Stored};
pub use adapter::{Control, JournalingAdapter};
pub use chain::{chain_start, chain_step, sha256_hex};
pub use record::{kind, source, Record};
pub use sink::{FileSink, Fsync, Sink};
pub use value::Value;
