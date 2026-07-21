//! Market data: bar aggregation specs, streaming builders, event streams,
//! and the deterministic multi-stream merge.

pub mod aggregate;
pub mod bar_spec;
pub mod events;
pub mod feed;

pub use aggregate::{builder_for, BarBuilder, SourceRecord, IST_OFFSET_NS};
pub use bar_spec::{AggregationUnit, BarSpec, SpecError};
pub use events::{tick_data_to_events, EventPayload, MarketEvent, QuoteTick, TradeTick};
pub use feed::EventFeed;
